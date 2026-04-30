from pathlib import Path

from mechrep.training.run_original_biopathnet_linux import (
    original_biopathnet_root,
    original_run_script,
    repository_root,
)


def test_original_biopathnet_linux_paths_exist():
    root = repository_root()

    assert root.name == "BPNVer"
    assert original_biopathnet_root(root) == root / "biopathnet" / "original"
    assert original_biopathnet_root(root).exists()
    assert original_run_script(root) == root / "biopathnet" / "original" / "script" / "run.py"
    assert original_run_script(root).exists()
    assert isinstance(root, Path)
