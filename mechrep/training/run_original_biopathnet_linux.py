"""Linux launcher for the original BioPathNet training script.

This entrypoint intentionally does not install any Windows fallback sparse
operators. It prepares import paths and the Torch extension cache, then delegates
to ``biopathnet/original/script/run.py`` unchanged.
"""

from __future__ import annotations

import os
import math
import pprint
import random
import runpy
import sys
from pathlib import Path
from typing import Any

import yaml


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def original_biopathnet_root(root: Path | None = None) -> Path:
    root = repository_root() if root is None else root
    return root / "biopathnet" / "original"


def original_run_script(root: Path | None = None) -> Path:
    return original_biopathnet_root(root) / "script" / "run.py"


def prepare_environment(root: Path | None = None) -> Path:
    """Prepare import paths for original BioPathNet without fallback patches."""

    root = repository_root() if root is None else root
    original_root = original_biopathnet_root(root)
    if not original_root.exists():
        raise FileNotFoundError(f"Original BioPathNet directory not found: {original_root}")

    extension_dir = root / ".torch_extensions_linux"
    extension_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extension_dir))

    original_root_str = str(original_root)
    if original_root_str not in sys.path:
        sys.path.insert(0, original_root_str)

    _set_cuda_arch_if_unset()
    _patch_torchdrug_extension_compatibility()

    from torch.utils import cpp_extension  # noqa: F401

    return original_root


def _set_cuda_arch_if_unset() -> None:
    """Avoid compiling extensions for irrelevant visible GPU architectures."""

    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return

    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability(0)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def _patch_torchdrug_extension_compatibility() -> None:
    """Patch TorchDrug's JIT loader for newer PyTorch releases.

    TorchDrug 0.2.1 calls ``torch.utils.cpp_extension.load`` with positional
    arguments. Newer PyTorch releases added parameters in the middle of that
    signature, which shifts ``build_directory`` into ``extra_include_paths`` and
    produces one ``-I`` flag per path character. Using keyword arguments keeps
    the original extension sources but makes the call robust across PyTorch
    versions.

    PyTorch 2.8 also ships sparse tensor helper headers under
    ``ATen/native/SparseTensorUtils.h`` instead of the older exported
    ``ATen/SparseTensorUtils.h`` path used by TorchDrug 0.2.1. The source patch
    below only rewrites the include when the old header is unavailable.
    """

    try:
        import torch
        from torch.utils import cpp_extension
        import torchdrug
        import torchdrug.utils.torch as torchdrug_torch
    except Exception:
        return

    def module(self):
        cached = self.__dict__.get("_mechrep_loaded_module")
        if cached is not None:
            return cached
        loaded = cpp_extension.load(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_paths=self.extra_include_paths,
            build_directory=self.build_directory,
            verbose=self.verbose,
            **self.kwargs,
        )
        self.__dict__["_mechrep_loaded_module"] = loaded
        return loaded

    torchdrug_torch.LazyExtensionLoader.module = property(module)

    torch_include = Path(torch.__file__).resolve().parent / "include"
    old_header = torch_include / "ATen" / "SparseTensorUtils.h"
    new_header = torch_include / "ATen" / "native" / "SparseTensorUtils.h"
    if old_header.exists() or not new_header.exists():
        return

    extension_dir = Path(torchdrug.__file__).resolve().parent / "layers" / "functional" / "extension"
    for header in [extension_dir / "spmm.h", extension_dir / "rspmm.h"]:
        if not header.exists():
            continue
        text = header.read_text(encoding="utf-8")
        patched = text.replace(
            "#include <ATen/SparseTensorUtils.h>",
            "#include <ATen/native/SparseTensorUtils.h>",
        )
        if patched != text:
            header.write_text(patched, encoding="utf-8")


def _config_path_from_argv(argv: list[str]) -> Path | None:
    for index, value in enumerate(argv):
        if value in {"-c", "--config"} and index + 1 < len(argv):
            return Path(argv[index + 1])
        if value.startswith("--config="):
            return Path(value.split("=", 1)[1])
    return None


def _runtime_options_from_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        return {}
    runtime = payload.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise ValueError("runtime config must be a mapping")
    return runtime


def _uses_runtime_controls(runtime: dict[str, Any]) -> bool:
    return (
        bool(runtime.get("skip_eval", False))
        or runtime.get("eval_num_negative") is not None
        or runtime.get("progress_bar") is not None
        or runtime.get("progress_log_interval") is not None
        or runtime.get("early_stop_patience") is not None
    )


def _patch_torchdrug_engine_progress(*, progress_bar: bool, log_interval_batches: int | None) -> None:
    """Add optional tqdm and line-based progress logging to TorchDrug Engine.train."""

    if not progress_bar and not log_interval_batches:
        return

    try:
        import logging
        import torch
        from itertools import islice
        from torch import nn
        from torch.utils import data as torch_data
        from torchdrug import data, utils
        from torchdrug.core.engine import Engine
        from torchdrug.utils import comm
    except Exception:
        return

    tqdm = None
    if progress_bar:
        try:
            from tqdm.auto import tqdm as tqdm
        except ImportError:
            tqdm = None

    if getattr(Engine.train, "_mechrep_tqdm_patch", False):
        return

    progress_logger = logging.getLogger(__name__)
    log_interval_batches = int(log_interval_batches or 0)

    def _metric_postfix(metric: dict[str, Any]) -> dict[str, float]:
        postfix = {}
        for key, value in list(metric.items())[:3]:
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if value.numel() != 1:
                    continue
                value = value.item()
            try:
                postfix[key] = float(value)
            except (TypeError, ValueError):
                continue
        return postfix

    def train(self, num_epoch=1, batch_per_epoch=None):
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        batch_per_epoch_resolved = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.device],
                    find_unused_parameters=True,
                )
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()

        train_start_epoch = self.epoch
        train_total_epoch = train_start_epoch + num_epoch

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            gradient_interval = min(batch_per_epoch_resolved - start_id, self.gradient_interval)
            batches = islice(dataloader, batch_per_epoch_resolved)
            if self.rank == 0 and tqdm is not None:
                batches = tqdm(
                    batches,
                    total=batch_per_epoch_resolved,
                    desc=f"BioPathNet epoch {self.epoch + 1}",
                    unit="batch",
                    dynamic_ncols=True,
                )

            for batch_id, batch in enumerate(batches):
                current_batch = batch_id + 1
                if (
                    self.rank == 0
                    and log_interval_batches > 0
                    and (
                        current_batch == 1
                        or current_batch % log_interval_batches == 0
                        or current_batch == batch_per_epoch_resolved
                    )
                ):
                    percent = current_batch / batch_per_epoch_resolved * 100
                    progress_logger.warning(
                        "BioPathNet progress: epoch %d/%d batch %d/%d (%.1f%%)",
                        self.epoch + 1,
                        train_total_epoch,
                        current_batch,
                        batch_per_epoch_resolved,
                        percent,
                    )

                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)
                    if self.rank == 0 and hasattr(batches, "set_postfix"):
                        batches.set_postfix(_metric_postfix(metric))

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch_resolved - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()

    train._mechrep_tqdm_patch = True
    Engine.train = train


def _solver_load_without_graphs(solver, checkpoint: str, *, load_optimizer: bool = True) -> None:
    import torch
    from torchdrug.utils import comm

    checkpoint = os.path.expanduser(checkpoint)
    state = torch.load(checkpoint, map_location=solver.device)
    for key in [
        "fact_graph",
        "fact_graph_supervision",
        "graph",
        "train_graph",
        "valid_graph",
        "test_graph",
        "full_valid_graph",
        "full_test_graph",
    ]:
        state["model"].pop(key, None)
    solver.model.load_state_dict(state["model"], strict=False)

    if load_optimizer:
        solver.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in solver.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(solver.device)
    comm.synchronize()


def _evaluate_with_runtime_controls(solver, split: str, *, eval_num_negative: int | None, logger):
    old_num_negative = getattr(solver.model, "num_negative", None)
    if eval_num_negative is not None:
        if eval_num_negative <= 0:
            raise ValueError("runtime.eval_num_negative must be positive")
        logger.warning("Runtime eval_num_negative: %d", eval_num_negative)
        solver.model.num_negative = int(eval_num_negative)
    try:
        return solver.evaluate(split)
    finally:
        if eval_num_negative is not None and old_num_negative is not None:
            solver.model.num_negative = old_num_negative


def _train_and_validate_with_runtime_controls(
    cfg,
    solver,
    *,
    skip_eval: bool,
    eval_num_negative: int | None,
    early_stop_patience: int | None,
    early_stop_min_delta: float,
    logger,
):
    if cfg.train.num_epoch == 0:
        return solver

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1
    bad_validations = 0

    for index in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - index)
        solver.model.split = "train"
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        if skip_eval:
            logger.warning("Runtime skip_eval=true: skip validation after epoch %d", solver.epoch)
            continue
        solver.model.split = "valid"
        metric = _evaluate_with_runtime_controls(
            solver,
            "valid",
            eval_num_negative=eval_num_negative,
            logger=logger,
        )
        result = metric[cfg.metric]
        if result > best_result + early_stop_min_delta:
            best_result = result
            best_epoch = solver.epoch
            bad_validations = 0
        else:
            bad_validations += 1
            if early_stop_patience is not None and bad_validations >= early_stop_patience:
                logger.warning(
                    "Runtime early stopping: no %s improvement for %d validation checks; best epoch %d",
                    cfg.metric,
                    bad_validations,
                    best_epoch,
                )
                break

    if not skip_eval and best_epoch >= 0:
        _solver_load_without_graphs(solver, "model_epoch_%d.pth" % best_epoch)
    return solver


def _test_with_runtime_controls(solver, *, skip_eval: bool, eval_num_negative: int | None, logger) -> None:
    if skip_eval:
        logger.warning("Runtime skip_eval=true: skip final valid/test evaluation")
        return
    solver.model.split = "valid"
    _evaluate_with_runtime_controls(solver, "valid", eval_num_negative=eval_num_negative, logger=logger)
    solver.model.split = "test"
    _evaluate_with_runtime_controls(solver, "test", eval_num_negative=eval_num_negative, logger=logger)


def _run_original_with_runtime_controls(runtime: dict[str, Any]) -> None:
    import numpy as np
    import torch
    from torchdrug import core, models as _torchdrug_models  # noqa: F401
    from torchdrug.utils import comm

    # Import BioPathNet modules for TorchDrug registry side effects. The
    # original run.py does this before loading config-defined classes.
    from biopathnet import dataset as _dataset  # noqa: F401
    from biopathnet import layer as _layer  # noqa: F401
    from biopathnet import model as _model  # noqa: F401
    from biopathnet import task as _task  # noqa: F401
    from biopathnet import util

    args, variables = util.parse_args()
    cfg = util.load_config(args.config, context=variables)
    working_dir = util.create_working_directory(cfg)
    args.seed = int(args.seed)
    seed_rank = args.seed + int(comm.get_rank())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed_rank)

    logger = util.get_root_logger()
    logger.warning("Working directory: %s" % working_dir)
    logger.warning("Input Seed: %d" % args.seed)
    logger.warning("Set Seed: %d" % seed_rank)
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Runtime controls: %s" % pprint.pformat(runtime))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    _patch_torchdrug_engine_progress(
        progress_bar=bool(runtime.get("progress_bar", True)),
        log_interval_batches=runtime.get("progress_log_interval", 100),
    )
    solver = util.build_solver(cfg, dataset)

    skip_eval = bool(runtime.get("skip_eval", False))
    eval_num_negative = runtime.get("eval_num_negative")
    if eval_num_negative is not None:
        eval_num_negative = int(eval_num_negative)
    early_stop_patience = runtime.get("early_stop_patience")
    if early_stop_patience is not None:
        early_stop_patience = int(early_stop_patience)
    early_stop_min_delta = float(runtime.get("early_stop_min_delta", 0.0))

    _train_and_validate_with_runtime_controls(
        cfg,
        solver,
        skip_eval=skip_eval,
        eval_num_negative=eval_num_negative,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        logger=logger,
    )
    _test_with_runtime_controls(
        solver,
        skip_eval=skip_eval,
        eval_num_negative=eval_num_negative,
        logger=logger,
    )


def main() -> None:
    root = repository_root()
    prepare_environment(root)
    run_script = original_run_script(root)
    if not run_script.exists():
        raise FileNotFoundError(f"Original BioPathNet run script not found: {run_script}")
    runtime = _runtime_options_from_config(_config_path_from_argv(sys.argv[1:]))
    if _uses_runtime_controls(runtime):
        _run_original_with_runtime_controls(runtime)
        return
    sys.argv[0] = str(run_script)
    runpy.run_path(str(run_script), run_name="__main__")


if __name__ == "__main__":
    main()
