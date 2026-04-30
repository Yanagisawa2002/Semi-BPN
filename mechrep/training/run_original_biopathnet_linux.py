"""Linux launcher for the original BioPathNet training script.

This entrypoint intentionally does not install any Windows fallback sparse
operators. It prepares import paths and the Torch extension cache, then delegates
to ``biopathnet/original/script/run.py`` unchanged.
"""

from __future__ import annotations

import os
import runpy
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
        cached = getattr(self, "_mechrep_loaded_module", None)
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
        self._mechrep_loaded_module = loaded
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
