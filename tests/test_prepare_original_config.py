from pathlib import Path

import yaml
import mechrep

from mechrep.training.prepare_original_config import materialize_config


def test_materialize_config_absolutizes_biopathnet_paths():
    repo_root = Path(mechrep.__file__).resolve().parents[1]
    root_path = repo_root / "results" / "test_tmp" / "prepare_original_config" / "project"
    root_path.mkdir(parents=True, exist_ok=True)
    config_path = root_path / "configs" / "smoke.yaml"
    output_path = root_path / "results" / "runtime_configs" / "smoke.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "output_dir: results/smoke",
                "dataset:",
                "  class: biomedical",
                "  path: data/cloud_run/biopathnet_full",
                "  files: ['train1.txt']",
                "runtime:",
                "  pairwise_eval:",
                "    enabled: true",
                "    split_dir: data/cloud_run/splits",
                "    output_dir: results/smoke_pairwise",
                "pairwise_training:",
                "  split_dir: data/cloud_run/splits",
                "  output_dir: results/pairwise_training",
            ]
        ),
        encoding="utf-8",
    )

    config = materialize_config(config_path, output_path, root_path)

    assert config["dataset"]["path"] == str(
        root_path / "data" / "cloud_run" / "biopathnet_full"
    )
    assert config["output_dir"] == str(root_path / "results" / "smoke")
    assert config["runtime"]["pairwise_eval"]["split_dir"] == str(root_path / "data" / "cloud_run" / "splits")
    assert config["runtime"]["pairwise_eval"]["output_dir"] == str(root_path / "results" / "smoke_pairwise")
    assert config["pairwise_training"]["split_dir"] == str(root_path / "data" / "cloud_run" / "splits")
    assert config["pairwise_training"]["output_dir"] == str(root_path / "results" / "pairwise_training")

    saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert saved == config
