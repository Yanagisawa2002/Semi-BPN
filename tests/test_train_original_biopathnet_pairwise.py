import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.build_pairs import PairRecord
from mechrep.training.train_original_biopathnet_pairwise import (
    batch_records,
    pairwise_options_from_config,
    should_validate_epoch,
)


def test_pairwise_options_from_config_reads_explicit_training_block():
    config = {
        "pairwise_training": {
            "split_dir": "data/cloud_run/splits",
            "relation_name": "indication",
            "train_batch_size": 8,
            "eval_batch_size": 16,
            "validation_interval": 3,
            "selection_metric": "auprc",
            "group_by": "endpoint_id",
            "k_values": [1, 5],
            "progress_bar": False,
            "progress_log_interval": 50,
            "early_stop_patience": 2,
            "early_stop_min_delta": 0.01,
            "shuffle": False,
            "final_splits": ["valid", "test"],
        }
    }

    options = pairwise_options_from_config(config)

    assert options.split_dir == Path("data/cloud_run/splits")
    assert options.relation_name == "indication"
    assert options.train_batch_size == 8
    assert options.eval_batch_size == 16
    assert options.validation_interval == 3
    assert options.selection_metric == "auprc"
    assert options.group_by == "endpoint_id"
    assert options.k_values == (1, 5)
    assert options.progress_bar is False
    assert options.progress_log_interval == 50
    assert options.early_stop_patience == 2
    assert options.early_stop_min_delta == 0.01
    assert options.shuffle is False
    assert options.final_splits == ("valid", "test")


def test_batch_records_is_deterministic_and_keeps_tail_batch():
    records = [
        PairRecord(pair_id=f"p{i}", drug_id=f"d{i}", endpoint_id=f"e{i}", label=i % 2)
        for i in range(5)
    ]

    batches_a = batch_records(records, batch_size=2, shuffle=True, seed=7, epoch=1)
    batches_b = batch_records(records, batch_size=2, shuffle=True, seed=7, epoch=1)
    batches_c = batch_records(records, batch_size=2, shuffle=False, seed=7, epoch=1)

    assert [[record.pair_id for record in batch] for batch in batches_a] == [
        [record.pair_id for record in batch] for batch in batches_b
    ]
    assert sum(len(batch) for batch in batches_a) == 5
    assert [len(batch) for batch in batches_a][-1] == 1
    assert [[record.pair_id for record in batch] for batch in batches_c] == [["p0", "p1"], ["p2", "p3"], ["p4"]]


def test_should_validate_epoch_always_validates_final_epoch():
    assert should_validate_epoch(3, num_epoch=9, validation_interval=3)
    assert not should_validate_epoch(4, num_epoch=9, validation_interval=3)
    assert should_validate_epoch(8, num_epoch=8, validation_interval=3)


def test_pairwise_training_config_defaults_are_for_indication_debug():
    config = yaml.safe_load(Path("configs/biopathnet_linux_full_indication_pairwise.yaml").read_text(encoding="utf-8"))

    assert config["dataset"]["path"] == "data/cloud_run/biopathnet_full_indication_debug"
    assert config["train"]["num_epoch"] == 9
    pairwise = config["pairwise_training"]
    assert pairwise["split_dir"] == "data/cloud_run/splits"
    assert pairwise["relation_name"] == "indication"
    assert pairwise["train_batch_size"] == 4
    assert pairwise["eval_batch_size"] == 16
    assert pairwise["validation_interval"] == 3
    assert pairwise["selection_metric"] == "auprc"
    assert pairwise["final_splits"] == ["train", "valid", "test"]
