import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.build_ood_splits import build_ood_splits


def _write_toy_pairs(path):
    rows = [
        ("p0", "d0", "e0", "1"),
        ("p1", "d1", "e0", "0"),
        ("p2", "d0", "e1", "1"),
        ("p3", "d1", "e1", "0"),
        ("p4", "d0", "e2", "1"),
        ("p5", "d1", "e2", "0"),
        ("p6", "d0", "e3", "1"),
        ("p7", "d1", "e3", "0"),
        ("p8", "d0", "e4", "1"),
        ("p9", "d1", "e4", "0"),
        ("p10", "d0", "e5", "1"),
        ("p11", "d1", "e5", "0"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["pair_id", "drug_id", "endpoint_id", "label"])
        writer.writerows(rows)
    return {row[0] for row in rows}


def _read_split(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _endpoint_ids(rows):
    return {row["endpoint_id"] for row in rows}


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"ood_split_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_endpoint_ood_split_has_no_train_test_endpoint_overlap():
    tmp_dir = _workspace_tmp_dir()
    try:
        input_tsv = tmp_dir / "pairs.tsv"
        all_pair_ids = _write_toy_pairs(input_tsv)
        output_dir = tmp_dir / "splits"

        report = build_ood_splits(
            input_tsv,
            output_dir,
            split_type="endpoint_ood",
            seed=123,
            valid_ratio=1 / 3,
            test_ratio=1 / 3,
        )

        train = _read_split(output_dir / "train.tsv")
        valid = _read_split(output_dir / "valid.tsv")
        test = _read_split(output_dir / "test.tsv")

        train_endpoints = _endpoint_ids(train)
        valid_endpoints = _endpoint_ids(valid)
        test_endpoints = _endpoint_ids(test)

        assert train_endpoints.isdisjoint(test_endpoints)
        assert train_endpoints.isdisjoint(valid_endpoints)
        assert valid_endpoints.isdisjoint(test_endpoints)
        assert train and valid and test

        written_pair_ids = {row["pair_id"] for row in train + valid + test}
        assert written_pair_ids == all_pair_ids

        with (output_dir / "leakage_report.json").open("r", encoding="utf-8") as handle:
            saved_report = json.load(handle)

        assert saved_report == report
        assert saved_report["number_of_train_endpoints"] == len(train_endpoints)
        assert saved_report["number_of_valid_endpoints"] == len(valid_endpoints)
        assert saved_report["number_of_test_endpoints"] == len(test_endpoints)
        assert saved_report["train_test_endpoint_overlap"] == 0
        assert saved_report["train_valid_endpoint_overlap"] == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_endpoint_ood_split_is_deterministic_under_fixed_seed():
    tmp_dir = _workspace_tmp_dir()
    try:
        input_tsv = tmp_dir / "pairs.tsv"
        _write_toy_pairs(input_tsv)

        output_a = tmp_dir / "split_a"
        output_b = tmp_dir / "split_b"

        build_ood_splits(input_tsv, output_a, seed=77, valid_ratio=1 / 3, test_ratio=1 / 3)
        build_ood_splits(input_tsv, output_b, seed=77, valid_ratio=1 / 3, test_ratio=1 / 3)

        for filename in ["train.tsv", "valid.tsv", "test.tsv", "leakage_report.json"]:
            assert (output_a / filename).read_text(encoding="utf-8") == (
                output_b / filename
            ).read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
