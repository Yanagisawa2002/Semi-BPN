import csv
import shutil
import time
from pathlib import Path
import uuid

import pytest

from mechrep.data.build_pairs import PairRecord
from mechrep.evaluation.score_original_biopathnet_pairs import (
    find_latest_checkpoint,
    infer_relation_name,
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
