from pathlib import Path

import yaml


def test_indication_debug_config_uses_indication_relation_and_three_epoch_eval():
    config = yaml.safe_load(Path("configs/biopathnet_linux_full_indication_debug.yaml").read_text(encoding="utf-8"))

    assert config["dataset"]["path"] == "data/cloud_run/biopathnet_full_indication_debug"
    assert config["train"]["num_epoch"] == 3
    assert config["engine"]["batch_size"] == 4
    assert config["runtime"]["validation_interval"] == 3
    assert config["runtime"]["pairwise_eval"]["batch_size"] == 16
    assert config["runtime"]["pairwise_eval"]["relation_name"] == "indication"
    assert config["runtime"]["pairwise_eval"]["selection_metric"] == "auprc"


def test_indication_debug_prepare_config_remaps_only_supervision_files():
    config = yaml.safe_load(Path("configs/biopathnet_indication_debug_prepare.yaml").read_text(encoding="utf-8"))

    assert config["source_dir"] == "data/cloud_run/biopathnet_full"
    assert config["output_dir"] == "data/cloud_run/biopathnet_full_indication_debug"
    assert config["source_relation"] == "affects_endpoint"
    assert config["target_relation"] == "indication"
    assert config["target_files"] == ["train2.txt", "valid.txt", "test.txt"]
    assert config["require_source_relation"] is True
    assert config["forbid_source_relation_in_copied_files"] is True
    assert config["forbid_target_relation_in_copied_files"] is True
