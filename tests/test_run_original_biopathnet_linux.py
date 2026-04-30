from pathlib import Path
import shutil
import uuid

from mechrep.training.run_original_biopathnet_linux import (
    _config_path_from_argv,
    _evaluate_with_runtime_controls,
    _runtime_options_from_config,
    _uses_runtime_controls,
    original_biopathnet_root,
    original_run_script,
    repository_root,
)


def _workspace_tmp_dir():
    path = repository_root() / ".test_tmp" / f"linux_runner_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_original_biopathnet_linux_paths_exist():
    root = repository_root()

    assert root.name == "BPNVer"
    assert original_biopathnet_root(root) == root / "biopathnet" / "original"
    assert original_biopathnet_root(root).exists()
    assert original_run_script(root) == root / "biopathnet" / "original" / "script" / "run.py"
    assert original_run_script(root).exists()
    assert isinstance(root, Path)


def test_runtime_control_helpers_parse_skip_eval():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = tmp_dir / "runtime.yaml"
        config.write_text(
            "\n".join(
                [
                    "runtime:",
                    "  skip_eval: true",
                    "  eval_num_negative: 4096",
                    "  progress_bar: true",
                    "  progress_log_interval: 100",
                ]
            ),
            encoding="utf-8",
        )

        assert _config_path_from_argv(["-s", "42", "-c", str(config)]) == config
        runtime = _runtime_options_from_config(config)
        assert runtime == {
            "skip_eval": True,
            "eval_num_negative": 4096,
            "progress_bar": True,
            "progress_log_interval": 100,
        }
        assert _uses_runtime_controls(runtime)
        assert _uses_runtime_controls({"progress_log_interval": 100})
        assert not _uses_runtime_controls({})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_eval_num_negative_is_temporary():
    class Logger:
        def warning(self, *args):
            pass

    class Model:
        num_negative = 4

    class Solver:
        def __init__(self):
            self.model = Model()
            self.seen = None

        def evaluate(self, split):
            self.seen = (split, self.model.num_negative)
            return {"mrr": 0.0}

    solver = Solver()
    metric = _evaluate_with_runtime_controls(
        solver,
        "valid",
        eval_num_negative=4096,
        logger=Logger(),
    )

    assert metric == {"mrr": 0.0}
    assert solver.seen == ("valid", 4096)
    assert solver.model.num_negative == 4
