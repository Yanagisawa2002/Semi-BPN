import csv
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.templates.build_gold_template_labels import run_gold_template_label_generation
from mechrep.templates.extract_templates import make_template_id, read_templates, run_template_extraction


GOLD_HEADER = [
    "pair_id",
    "drug_id",
    "endpoint_id",
    "path_id",
    "path_node_ids",
    "path_node_types",
    "path_relation_types",
    "split",
]


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"labels_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_pair_split(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["pair_id", "drug_id", "endpoint_id", "label"])
        writer.writerows(rows)


def _write_pair_splits(pair_split_dir):
    _write_pair_split(
        pair_split_dir / "train.tsv",
        [
            ("P_train_1", "D1", "E_train", "1"),
            ("P_train_2", "D2", "E_train", "1"),
            ("P_train_3", "D3", "E_train", "1"),
        ],
    )
    _write_pair_split(pair_split_dir / "valid.tsv", [("P_valid_1", "D1", "E_valid", "1")])
    _write_pair_split(pair_split_dir / "test.tsv", [("P_test_1", "D1", "E_test", "1")])


def _write_gold_paths(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(GOLD_HEADER)
        writer.writerows(rows)


def _read_labels(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _config(tmp_dir, *, extract_from="train_only"):
    return {
        "gold_path_file": str(tmp_dir / "gold_paths.tsv"),
        "pair_split_dir": str(tmp_dir / "splits"),
        "output_dir": str(tmp_dir / "out"),
        "min_support": 1,
        "extract_from": extract_from,
        "delimiter": "\t",
        "sequence_separator": "|",
    }


def test_gold_template_labels_handle_multiple_templates_and_primary_tie_breaking():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _config(tmp_dir)
        _write_pair_splits(Path(config["pair_split_dir"]))
        _write_gold_paths(
            Path(config["gold_path_file"]),
            [
                ("P_train_1", "D1", "E_train", "path_a1", "D1|G1|E_train", "Drug|Gene|Endpoint", "targets|associated_with", "train"),
                ("P_train_1", "D1", "E_train", "path_b1", "D1|PW1|E_train", "Drug|Pathway|Endpoint", "participates_in|associated_with", "train"),
                ("P_train_2", "D2", "E_train", "path_a2", "D2|G2|E_train", "Drug|Gene|Endpoint", "targets|associated_with", "train"),
                ("P_train_3", "D3", "E_train", "path_b2", "D3|PW2|E_train", "Drug|Pathway|Endpoint", "participates_in|associated_with", "train"),
            ],
        )

        run_template_extraction(config)
        labels = run_gold_template_label_generation(config)
        rows = _read_labels(Path(config["output_dir"]) / "gold_template_labels_train.tsv")

        template_a = make_template_id("Drug|Gene|Endpoint", "targets|associated_with")
        template_b = make_template_id("Drug|Pathway|Endpoint", "participates_in|associated_with")
        expected_primary = sorted([template_a, template_b])[0]

        label_by_pair = {row["pair_id"]: row for row in rows}
        assert label_by_pair["P_train_1"]["template_ids"] == "|".join(sorted([template_a, template_b]))
        assert label_by_pair["P_train_1"]["primary_template_id"] == expected_primary
        assert label_by_pair["P_train_1"]["num_gold_paths"] == "2"
        assert any(label.pair_id == "P_train_1" for label in labels)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_primary_template_prefers_highest_support_before_lexical_order():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _config(tmp_dir)
        _write_pair_splits(Path(config["pair_split_dir"]))
        _write_gold_paths(
            Path(config["gold_path_file"]),
            [
                ("P_train_1", "D1", "E_train", "path_low", "D1|A1|E_train", "Drug|Anatomy|Endpoint", "localizes|associated_with", "train"),
                ("P_train_1", "D1", "E_train", "path_high", "D1|G1|E_train", "Drug|Gene|Endpoint", "targets|associated_with", "train"),
                ("P_train_2", "D2", "E_train", "path_high_2", "D2|G2|E_train", "Drug|Gene|Endpoint", "targets|associated_with", "train"),
            ],
        )

        run_template_extraction(config)
        run_gold_template_label_generation(config)
        row = {
            item["pair_id"]: item
            for item in _read_labels(Path(config["output_dir"]) / "gold_template_labels_train.tsv")
        }["P_train_1"]

        high_support_template = make_template_id("Drug|Gene|Endpoint", "targets|associated_with")
        assert row["primary_template_id"] == high_support_template
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_test_gold_paths_do_not_define_train_only_templates_but_files_are_written():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _config(tmp_dir, extract_from="train_only")
        _write_pair_splits(Path(config["pair_split_dir"]))
        _write_gold_paths(
            Path(config["gold_path_file"]),
            [
                ("P_train_1", "D1", "E_train", "path_train", "D1|G1|E_train", "Drug|Gene|Endpoint", "targets|associated_with", "train"),
                ("P_test_1", "D1", "E_test", "path_test_unique", "D1|PW1|E_test", "Drug|Pathway|Endpoint", "participates_in|associated_with", "test"),
            ],
        )

        run_template_extraction(config)
        run_gold_template_label_generation(config)

        templates = read_templates(Path(config["output_dir"]) / "templates.tsv")
        test_unique = make_template_id("Drug|Pathway|Endpoint", "participates_in|associated_with")
        assert test_unique not in {template.template_id for template in templates}
        assert (Path(config["output_dir"]) / "gold_template_labels_test.tsv").exists()
        assert _read_labels(Path(config["output_dir"]) / "gold_template_labels_test.tsv") == []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_label_generation_requires_gold_pair_id_in_matching_split():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _config(tmp_dir)
        _write_pair_splits(Path(config["pair_split_dir"]))
        _write_gold_paths(
            Path(config["gold_path_file"]),
            [
                ("P_missing", "D1", "E_train", "path_missing", "D1|G1|E_train", "Drug|Gene|Endpoint", "targets|associated_with", "train")
            ],
        )

        with pytest.raises(ValueError, match="not found in the corresponding pair split file"):
            run_gold_template_label_generation(config)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
