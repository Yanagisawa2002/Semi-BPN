"""Train original BioPathNet with explicit endpoint pair labels.

This trainer keeps the original BioPathNet / NBFNet encoder intact, but replaces
BioPathNet's internal random corruption objective with direct BCE on our
positive and negative endpoint pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pprint
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from mechrep.data.build_pairs import PairRecord, read_pair_tsv
from mechrep.evaluation.eval_prediction import evaluate_predictions, write_metrics_json
from mechrep.evaluation.score_original_biopathnet_pairs import _write_predictions, score_records
from mechrep.training.progress import EarlyStopper, iter_progress
from mechrep.training.run_original_biopathnet_linux import (
    _solver_load_without_graphs,
    prepare_environment,
    repository_root,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairwiseTrainingOptions:
    split_dir: Path
    relation_name: str
    train_batch_size: int
    eval_batch_size: int
    validation_interval: int
    selection_metric: str
    group_by: str | None
    k_values: tuple[int, ...]
    progress_bar: bool
    progress_log_interval: int
    early_stop_patience: int | None
    early_stop_min_delta: float
    shuffle: bool
    final_splits: tuple[str, ...]


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value


def _mapping_get(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    return getattr(config, key, default)


def _as_path(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser()
    if not str(path):
        raise ValueError(f"{name} must be non-empty")
    return path


def _parse_group_by(value: Any) -> str | None:
    if value in (None, "", "none", "None"):
        return None
    return str(value)


def _parse_splits(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("train", "valid", "test")
    if isinstance(value, str):
        splits = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        splits = tuple(str(part).strip() for part in value if str(part).strip())
    allowed = {"train", "valid", "test"}
    unknown = sorted(set(splits) - allowed)
    if unknown:
        raise ValueError(f"final_splits contains unknown splits: {unknown}")
    if not splits:
        raise ValueError("final_splits must contain at least one split")
    return splits


def pairwise_options_from_config(config: Mapping[str, Any]) -> PairwiseTrainingOptions:
    pairwise = _mapping_get(config, "pairwise_training", {}) or {}
    runtime = _mapping_get(config, "runtime", {}) or {}
    pairwise_eval = _mapping_get(runtime, "pairwise_eval", {}) or {}

    split_dir = _mapping_get(pairwise, "split_dir", _mapping_get(pairwise_eval, "split_dir"))
    if split_dir is None:
        raise ValueError("pairwise_training.split_dir is required")

    relation_name = str(_mapping_get(pairwise, "relation_name", _mapping_get(pairwise_eval, "relation_name", "")))
    if not relation_name:
        raise ValueError("pairwise_training.relation_name is required")

    train_batch_size = int(_mapping_get(pairwise, "train_batch_size", _mapping_get(pairwise, "batch_size", 4)))
    eval_batch_size = int(_mapping_get(pairwise, "eval_batch_size", _mapping_get(pairwise_eval, "batch_size", 16)))
    validation_interval = int(_mapping_get(pairwise, "validation_interval", _mapping_get(runtime, "validation_interval", 3)))
    progress_log_interval = int(_mapping_get(pairwise, "progress_log_interval", _mapping_get(runtime, "progress_log_interval", 100)))
    early_stop_patience = _mapping_get(pairwise, "early_stop_patience", _mapping_get(runtime, "early_stop_patience"))
    if early_stop_patience in ("", "none", "None"):
        early_stop_patience = None

    k_values = tuple(int(value) for value in _mapping_get(pairwise, "k_values", _mapping_get(pairwise_eval, "k_values", [1, 5, 10])))
    if train_batch_size <= 0:
        raise ValueError(f"train_batch_size must be positive, got {train_batch_size}")
    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}")
    if validation_interval <= 0:
        raise ValueError(f"validation_interval must be positive, got {validation_interval}")
    if progress_log_interval < 0:
        raise ValueError(f"progress_log_interval must be non-negative, got {progress_log_interval}")
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError(f"k_values must be positive, got {k_values}")

    return PairwiseTrainingOptions(
        split_dir=_as_path(split_dir, name="pairwise_training.split_dir"),
        relation_name=relation_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        validation_interval=validation_interval,
        selection_metric=str(_mapping_get(pairwise, "selection_metric", _mapping_get(pairwise_eval, "selection_metric", "auprc"))),
        group_by=_parse_group_by(_mapping_get(pairwise, "group_by", _mapping_get(pairwise_eval, "group_by", "endpoint_id"))),
        k_values=k_values,
        progress_bar=bool(_mapping_get(pairwise, "progress_bar", _mapping_get(runtime, "progress_bar", True))),
        progress_log_interval=progress_log_interval,
        early_stop_patience=int(early_stop_patience) if early_stop_patience is not None else None,
        early_stop_min_delta=float(_mapping_get(pairwise, "early_stop_min_delta", _mapping_get(runtime, "early_stop_min_delta", 0.0))),
        shuffle=bool(_mapping_get(pairwise, "shuffle", True)),
        final_splits=_parse_splits(_mapping_get(pairwise, "final_splits", ("train", "valid", "test"))),
    )


def batch_records(
    records: Sequence[PairRecord],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[list[PairRecord]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    ordered = list(records)
    if shuffle:
        rng = random.Random(seed + epoch)
        rng.shuffle(ordered)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def should_validate_epoch(epoch: int, *, num_epoch: int, validation_interval: int) -> bool:
    if epoch <= 0 or num_epoch <= 0:
        return False
    return epoch % validation_interval == 0 or epoch == num_epoch


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_checkpoint(solver, path: str | Path, *, epoch: int, metrics: dict[str, Any] | None = None) -> None:
    import torch

    path = Path(path)
    state = {
        "model": solver.model.state_dict(),
        "optimizer": solver.optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
    }
    torch.save(state, path)


def _validate_relation_and_entities(records: Sequence[PairRecord], *, dataset, relation_name: str, split: str) -> int:
    if relation_name not in dataset.inv_relation_vocab:
        raise ValueError(f"Relation {relation_name!r} is missing from BioPathNet relation vocabulary")
    missing = []
    for record in records:
        if record.drug_id not in dataset.inv_entity_vocab:
            missing.append((record.pair_id, "drug_id", record.drug_id))
        if record.endpoint_id not in dataset.inv_entity_vocab:
            missing.append((record.pair_id, "endpoint_id", record.endpoint_id))
        if len(missing) >= 10:
            break
    if missing:
        raise ValueError(f"Split {split!r} contains pairs with entities missing from BioPathNet graph: {missing}")
    return int(dataset.inv_relation_vocab[relation_name])


def _pairwise_logits(records: Sequence[PairRecord], *, dataset, solver, relation_index: int):
    import torch

    device = solver.device
    h_index = torch.tensor(
        [dataset.inv_entity_vocab[record.drug_id] for record in records],
        dtype=torch.long,
        device=device,
    ).unsqueeze(-1)
    t_index = torch.tensor(
        [dataset.inv_entity_vocab[record.endpoint_id] for record in records],
        dtype=torch.long,
        device=device,
    ).unsqueeze(-1)
    r_index = torch.full((len(records), 1), relation_index, dtype=torch.long, device=device)
    return solver.model.model(solver.model.fact_graph, h_index, t_index, r_index).view(-1)


def _train_one_epoch(
    *,
    records: Sequence[PairRecord],
    dataset,
    solver,
    options: PairwiseTrainingOptions,
    epoch: int,
    num_epoch: int,
    seed: int,
    logger: logging.Logger,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    relation_index = _validate_relation_and_entities(
        records,
        dataset=dataset,
        relation_name=options.relation_name,
        split="train",
    )
    batches = batch_records(
        records,
        batch_size=options.train_batch_size,
        shuffle=options.shuffle,
        seed=seed,
        epoch=epoch,
    )
    iterator = iter_progress(
        enumerate(batches, start=1),
        enabled=options.progress_bar,
        total=len(batches),
        desc=f"BioPathNet pairwise epoch {epoch}",
        unit="batch",
    )

    solver.model.train()
    solver.model.split = "train"
    total_loss = 0.0
    total_examples = 0
    positive_score_sum = 0.0
    negative_score_sum = 0.0
    num_positive = 0
    num_negative = 0

    for batch_id, batch in iterator:
        labels = torch.tensor([record.label for record in batch], dtype=torch.float, device=solver.device)
        solver.optimizer.zero_grad()
        logits = _pairwise_logits(batch, dataset=dataset, solver=solver, relation_index=relation_index)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        solver.optimizer.step()

        batch_size = len(batch)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        with torch.no_grad():
            scores = torch.sigmoid(logits.detach())
            positive_mask = labels == 1
            negative_mask = labels == 0
            if bool(positive_mask.any()):
                positive_score_sum += float(scores[positive_mask].sum().detach().cpu())
                num_positive += int(positive_mask.sum().detach().cpu())
            if bool(negative_mask.any()):
                negative_score_sum += float(scores[negative_mask].sum().detach().cpu())
                num_negative += int(negative_mask.sum().detach().cpu())

        if (
            options.progress_log_interval
            and (
                batch_id == 1
                or batch_id % options.progress_log_interval == 0
                or batch_id == len(batches)
            )
        ):
            logger.warning(
                "BioPathNet pairwise progress: epoch %d/%d batch %d/%d (%.1f%%) loss %.5g",
                epoch,
                num_epoch,
                batch_id,
                len(batches),
                batch_id / len(batches) * 100,
                float(loss.detach().cpu()),
            )
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": float(loss.detach().cpu())})

    return {
        "epoch": float(epoch),
        "loss": total_loss / total_examples,
        "num_examples": float(total_examples),
        "num_positive": float(num_positive),
        "num_negative": float(num_negative),
        "mean_positive_score": positive_score_sum / num_positive if num_positive else math.nan,
        "mean_negative_score": negative_score_sum / num_negative if num_negative else math.nan,
    }


def _evaluate_pairwise_split(
    *,
    records: Sequence[PairRecord],
    dataset,
    solver,
    options: PairwiseTrainingOptions,
    split: str,
    output_dir: Path,
    suffix: str,
) -> dict[str, Any]:
    solver.model.eval()
    rows = score_records(
        records,
        dataset=dataset,
        solver=solver,
        relation_name=options.relation_name,
        split=split,
        batch_size=options.eval_batch_size,
        progress_bar=options.progress_bar,
    )
    metrics = evaluate_predictions(rows, k_values=options.k_values, group_by=options.group_by)
    metrics["num_examples"] = len(rows)
    metrics["num_positive"] = sum(int(row["label"]) for row in rows)
    metrics["num_negative"] = len(rows) - metrics["num_positive"]

    predictions_path = output_dir / f"predictions_{split}_{suffix}.tsv"
    metrics_path = output_dir / f"metrics_{split}_{suffix}.json"
    _write_predictions(predictions_path, rows)
    write_metrics_json(metrics_path, metrics)
    return {
        "split": split,
        "suffix": suffix,
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "metrics_summary": metrics,
    }


def _load_original_solver(config_path: Path, *, seed: int):
    import numpy as np
    import torch
    from torchdrug import core, models as _torchdrug_models  # noqa: F401
    from torchdrug.utils import comm

    from biopathnet import dataset as _dataset  # noqa: F401
    from biopathnet import layer as _layer  # noqa: F401
    from biopathnet import model as _model  # noqa: F401
    from biopathnet import task as _task  # noqa: F401
    from biopathnet import util

    args_seed = int(seed)
    seed_rank = args_seed + int(comm.get_rank())
    random.seed(args_seed)
    np.random.seed(args_seed)
    torch.manual_seed(seed_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_rank)

    cfg = util.load_config(str(config_path), context={})
    working_dir = Path(util.create_working_directory(cfg)).resolve()
    logger = util.get_root_logger()
    logger.warning("Working directory: %s", working_dir)
    logger.warning("Input Seed: %d", args_seed)
    logger.warning("Set Seed: %d", seed_rank)
    if comm.get_rank() == 0:
        logger.warning("Config file: %s", config_path)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    return cfg, dataset, solver, logger, working_dir


def run_pairwise_training(config_path: str | Path, *, seed: int = 42) -> dict[str, Any]:
    root = repository_root()
    prepare_environment(root)
    config_path = Path(config_path).resolve(strict=False)
    cfg, dataset, solver, logger, working_dir = _load_original_solver(config_path, seed=seed)
    options = pairwise_options_from_config(cfg)

    split_records = {
        split: read_pair_tsv(options.split_dir / f"{split}.tsv", split=split)
        for split in ("train", "valid", "test")
    }
    for split, records in split_records.items():
        _validate_relation_and_entities(records, dataset=dataset, relation_name=options.relation_name, split=split)

    shutil.copy2(config_path, working_dir / "config.yaml")
    _write_json(
        working_dir / "seed.json",
        {
            "seed": int(seed),
            "relation_name": options.relation_name,
            "split_dir": str(options.split_dir),
        },
    )

    num_epoch = int(cfg.train.num_epoch)
    if num_epoch <= 0:
        raise ValueError(f"train.num_epoch must be positive, got {num_epoch}")

    history = []
    validation_dir = working_dir / "pairwise_validation"
    final_dir = working_dir / "pairwise_final"
    checkpoints = []
    early_stopper = EarlyStopper(
        options.selection_metric,
        options.early_stop_patience,
        min_delta=options.early_stop_min_delta,
    )
    best_checkpoint: Path | None = None
    best_epoch = 0
    best_metric_value: float | None = None

    logger.warning("Explicit pairwise training options: %s", pprint.pformat(options))
    for epoch in range(1, num_epoch + 1):
        train_metrics = _train_one_epoch(
            records=split_records["train"],
            dataset=dataset,
            solver=solver,
            options=options,
            epoch=epoch,
            num_epoch=num_epoch,
            seed=seed,
            logger=logger,
        )
        checkpoint_path = working_dir / f"model_epoch_{epoch}.pth"
        _save_checkpoint(solver, checkpoint_path, epoch=epoch, metrics={"train": train_metrics})
        checkpoints.append(str(checkpoint_path))

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "train": train_metrics,
        }

        if should_validate_epoch(epoch, num_epoch=num_epoch, validation_interval=options.validation_interval):
            valid_report = _evaluate_pairwise_split(
                records=split_records["valid"],
                dataset=dataset,
                solver=solver,
                options=options,
                split="valid",
                output_dir=validation_dir,
                suffix=f"epoch_{epoch}",
            )
            valid_metrics = valid_report["metrics_summary"]
            epoch_record["valid"] = valid_report
            if options.selection_metric not in valid_metrics:
                raise ValueError(f"selection_metric {options.selection_metric!r} is unavailable")
            metric_value = float(valid_metrics[options.selection_metric])
            improved, should_stop = early_stopper.observe(metric_value, epoch=epoch)
            if improved or best_checkpoint is None:
                best_checkpoint = checkpoint_path
                best_epoch = epoch
                best_metric_value = metric_value
            logger.warning(
                "Explicit pairwise valid epoch %d: %s %.5g AUROC %.5g AUPRC %.5g",
                epoch,
                options.selection_metric,
                metric_value,
                valid_metrics["auroc"],
                valid_metrics["auprc"],
            )
            if should_stop:
                epoch_record["early_stop"] = True
                history.append(epoch_record)
                logger.warning(
                    "Explicit pairwise early stopping at epoch %d; best epoch %d",
                    epoch,
                    early_stopper.best_epoch,
                )
                break
        else:
            if best_checkpoint is None:
                best_checkpoint = checkpoint_path
                best_epoch = epoch

        history.append(epoch_record)
        _write_json(working_dir / "history.json", history)

    if best_checkpoint is None:
        best_checkpoint = Path(checkpoints[-1])
        best_epoch = int(history[-1]["epoch"])

    _solver_load_without_graphs(solver, str(best_checkpoint), load_optimizer=False)
    final_reports = {}
    for split in options.final_splits:
        final_reports[split] = _evaluate_pairwise_split(
            records=split_records[split],
            dataset=dataset,
            solver=solver,
            options=options,
            split=split,
            output_dir=final_dir,
            suffix=f"best_epoch_{best_epoch}",
        )

    summary = {
        "working_dir": str(working_dir),
        "config": str(config_path),
        "seed": int(seed),
        "relation_name": options.relation_name,
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": int(best_epoch),
        "best_selection_metric": options.selection_metric,
        "best_selection_metric_value": best_metric_value,
        "history": history,
        "final": final_reports,
        "checkpoints": checkpoints,
    }
    _write_json(working_dir / "summary.json", summary)
    _write_json(
        working_dir / "metrics.json",
        {
            split: report["metrics_summary"]
            for split, report in final_reports.items()
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-s", "--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pairwise_training(args.config, seed=args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
