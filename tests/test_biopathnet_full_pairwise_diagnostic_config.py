from pathlib import Path

import yaml


def test_full_pairwise_diagnostic_config_is_safe_by_default():
    config = yaml.safe_load(Path("configs/biopathnet_linux_full_pairwise_diagnostic.yaml").read_text(encoding="utf-8"))

    assert config["dataset"]["path"] == "data/cloud_run/biopathnet_full"
    assert config["train"]["num_epoch"] == 5
    assert config["engine"]["batch_size"] == 1
    assert config["task"]["model"]["input_dim"] == 16
    assert config["task"]["model"]["hidden_dims"] == [16, 16]

    runtime = config["runtime"]
    assert runtime["skip_eval"] is True
    assert runtime["skip_final_eval"] is True
    assert runtime["validation_interval"] == 1
    assert runtime["pairwise_eval"]["enabled"] is True
    assert runtime["pairwise_eval"]["batch_size"] == 1
    assert runtime["pairwise_eval"]["selection_metric"] == "auprc"
