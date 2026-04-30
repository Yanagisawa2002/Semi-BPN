import csv
import shutil
import time
from pathlib import Path
import uuid

import pytest

from mechrep.data.build_pairs import PairRecord
from mechrep.evaluation.score_original_biopathnet_pairs import (
    checkpoint_epoch,
    find_checkpoints,
    find_latest_checkpoint,
    infer_relation_name,
    normalize_selection_metric,
    selection_metric_mode,
    validate_records_mappable,
)


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"pair_score_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_find_latest_checkpoint_uses_mtime():
    tmp_dir = _workspace_tmp_dir()
    try:
        old = tmp_dir / "run_a" / "model_epoch_1.pth"
        new = tmp_dir / "run_b" / "model_epoch_2.pth"
        old.parent.mkdir()
        new.parent.mkdir()
        old.write_text("old", encoding="utf-8")
        time.sleep(0.01)
        new.write_text("new", encoding="utf-8")

        assert find_latest_checkpoint(tmp_dir) == new
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_find_checkpoints_sorts_by_epoch_before_mtime():
    tmp_dir = _workspace_tmp_dir()
    try:
        epoch_10 = tmp_dir / "run" / "model_epoch_10.pth"
        epoch_2 = tmp_dir / "run" / "model_epoch_2.pth"
        epoch_10.parent.mkdir()
        epoch_10.write_text("epoch 10", encoding="utf-8")
        time.sleep(0.01)
        epoch_2.write_text("epoch 2", encoding="utf-8")

        assert checkpoint_epoch(epoch_10) == 10
        assert find_checkpoints(tmp_dir) == [epoch_2, epoch_10]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_selection_metric_helpers_normalize_split_prefix_and_mode():
    assert normalize_selection_metric("valid_auprc") == "auprc"
    assert selection_metric_mode("valid_auprc") == "max"
    assert selection_metric_mode("valid_loss") == "min"


def test_infer_relation_name_from_train2():
    tmp_dir = _workspace_tmp_dir()
    try:
        graph_dir = tmp_dir / "graph"
        graph_dir.mkdir()
        with (graph_dir / "train2.txt").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["drug:a", "affects_endpoint", "disease:x"])

        assert infer_relation_name(graph_dir) == "affects_endpoint"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_validate_records_mappable_reports_missing_entities():
    records = [PairRecord("p0", "drug:a", "disease:x", 1, "valid")]
    with pytest.raises(ValueError, match="entities missing"):
        validate_records_mappable(records, {"drug:a": 0}, split="valid")
