"""Small launcher for the original BioPathNet training script.

This wrapper keeps the original implementation untouched. It only prepares the
runtime environment used by TorchDrug on Windows before delegating to
``biopathnet/original/script/run.py``.
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def original_biopathnet_root(root: Path | None = None) -> Path:
    root = repository_root() if root is None else root
    return root / "biopathnet" / "original"


def original_run_script(root: Path | None = None) -> Path:
    return original_biopathnet_root(root) / "script" / "run.py"


def prepare_environment(root: Path | None = None) -> Path:
    """Prepare import paths and extension cache for original BioPathNet."""

    root = repository_root() if root is None else root
    original_root = original_biopathnet_root(root)
    if not original_root.exists():
        raise FileNotFoundError(f"Original BioPathNet directory not found: {original_root}")

    extension_dir = root / ".torch_extensions"
    extension_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(extension_dir))

    original_root_str = str(original_root)
    if original_root_str not in sys.path:
        sys.path.insert(0, original_root_str)

    # TorchDrug lazily imports this inside its extension loader. In the local
    # Windows conda environment that lazy import can hang, while eager import is
    # fast and makes the original loader proceed normally.
    from torch.utils import cpp_extension  # noqa: F401
    _patch_torchdrug_extensions_for_windows()

    return original_root


def _windows_cflags(flags: list[str] | None) -> list[str] | None:
    if flags is None or os.name != "nt":
        return flags

    mapped = []
    for flag in flags:
        if flag == "-Ofast":
            mapped.append("/O2")
        elif flag == "-fopenmp":
            mapped.append("/openmp")
        elif flag == "-g":
            mapped.append("/Zi")
        else:
            mapped.append(flag)
    return mapped


def _patch_torchdrug_extensions_for_windows() -> None:
    if os.name != "nt":
        return

    cl = shutil.which("cl")
    if cl:
        os.environ.setdefault("CC", cl)
        os.environ.setdefault("CXX", cl)

    import torch
    from torch.utils import cpp_extension
    import torchdrug.utils.torch as torchdrug_torch

    def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
        if extra_cflags is None:
            extra_cflags = ["/O2"]
            if torch.backends.openmp.is_available():
                extra_cflags += ["/openmp", "-DAT_PARALLEL_OPENMP"]
            else:
                extra_cflags.append("-DAT_PARALLEL_NATIVE")
        else:
            extra_cflags = _windows_cflags(list(extra_cflags))

        if extra_cuda_cflags is None:
            if torch.cuda.is_available():
                extra_cuda_cflags = ["-O3"]
                extra_cflags.append("-DCUDA_OP")
            else:
                sources = [
                    source for source in sources if not cpp_extension._is_cuda_file(source)
                ]

        return torchdrug_torch.LazyExtensionLoader(
            name, sources, extra_cflags, extra_cuda_cflags, **kwargs
        )

    torchdrug_torch.load_extension = load_extension
    parent_utils = sys.modules.get("torchdrug.utils")
    if parent_utils is not None:
        parent_utils.load_extension = load_extension

    path = Path(torchdrug_torch.__file__).resolve().parent / "extension"
    torchdrug_torch.torch_ext = load_extension("torch_ext", [str(path / "torch_ext.cpp")])
    _patch_torchdrug_sparse_fallbacks_for_windows(torchdrug_torch, parent_utils)


def _patch_torchdrug_sparse_fallbacks_for_windows(torchdrug_torch, parent_utils) -> None:
    import torch
    from torch_scatter import scatter_add, scatter_max, scatter_min

    def sparse_coo_tensor(indices, values, size):
        return torch.sparse_coo_tensor(indices, values, size).coalesce()

    def generalized_rspmm(sparse, relation, input, sum="add", mul="mul"):
        sparse = sparse.coalesce()
        node_out, node_in, rel = sparse.indices()
        values = sparse.values()
        chunk_size = int(os.environ.get("BPN_RSPMM_CHUNK_SIZE", "250000"))
        num_node = sparse.size(0)
        width = relation.shape[-1]

        if sum == "add":
            output = input.new_zeros(num_node, width)
        elif sum == "max":
            output = input.new_full((num_node, width), -float("inf"))
        elif sum == "min":
            output = input.new_full((num_node, width), float("inf"))
        else:
            raise ValueError(f"Unsupported rspmm summation: {sum}")

        for start in range(0, node_out.numel(), chunk_size):
            end = min(start + chunk_size, node_out.numel())
            chunk_out = node_out[start:end]
            chunk_in = node_in[start:end]
            chunk_rel = rel[start:end]
            chunk_values = values[start:end].unsqueeze(-1)

            if mul == "mul":
                message = relation[chunk_rel] * input[chunk_in]
            elif mul == "add":
                message = relation[chunk_rel] + input[chunk_in]
            else:
                raise ValueError(f"Unsupported rspmm multiplication: {mul}")
            message = message * chunk_values

            if sum == "add":
                output = output + scatter_add(message, chunk_out, dim=0, dim_size=num_node)
            elif sum == "max":
                init = input.new_full((num_node, width), -float("inf"))
                chunk_result = scatter_max(message, chunk_out, dim=0, out=init)[0]
                output = torch.maximum(output, chunk_result)
            elif sum == "min":
                init = input.new_full((num_node, width), float("inf"))
                chunk_result = scatter_min(message, chunk_out, dim=0, out=init)[0]
                output = torch.minimum(output, chunk_result)
            del message

        if sum == "max":
            output = torch.nan_to_num(output, neginf=0.0)
        if sum == "max":
            return output
        if sum == "min":
            return torch.nan_to_num(output, posinf=0.0)
        return output

    torchdrug_torch.sparse_coo_tensor = sparse_coo_tensor
    if parent_utils is not None:
        parent_utils.sparse_coo_tensor = sparse_coo_tensor

    try:
        import torchdrug.layers.functional as functional
        import torchdrug.layers.functional.spmm as spmm_module
    except Exception:
        return

    spmm_module.generalized_rspmm = generalized_rspmm
    functional.generalized_rspmm = generalized_rspmm


def main() -> None:
    root = repository_root()
    prepare_environment(root)
    run_script = original_run_script(root)
    if not run_script.exists():
        raise FileNotFoundError(f"Original BioPathNet run script not found: {run_script}")
    sys.argv[0] = str(run_script)
    runpy.run_path(str(run_script), run_name="__main__")


if __name__ == "__main__":
    main()
