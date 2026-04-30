import csv
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.templates.extract_templates import (
    make_template_id,
    read_gold_paths,
    read_templates,
    run_template_extraction,
)


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
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"templates_{uuid.uuid4().hex}"
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


def _config(tmp_dir, *, min_support=1, extract_from="train_only"):
    return {
        "gold_path_file": str(tmp_dir / "gold_paths.tsv"),
        "pair_split_dir": str(tmp_dir / "splits"),
        "output_dir": str(tmp_dir / "out"),
        "min_support": min_support,
        "extract_from": extract_from,
        "delimiter": "\t",
        "sequence_separator": "|",
    }


def test_template_extraction_is_deterministic_and_counts_duplicate_support():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _config(tmp_dir)
        _write_pair_splits(Path(config["pair_split_dir"]))
        _write_gold_paths(
            Path(config["gold_path_file"]),
            [
                (
                    "P_train_1",
                    "D1",
                    "E_train",
                    "path_2",
                    "D1|G1|E_train",
                    "Drug|Gene|Endpoint",
                    "targets|associated_with",
                    "train",
                ),
                (
                    "P_train_2",
                    "D2",
                    "E_train",
                    "path_1",
                    "D2|G2|E_train",
                    "Drug|Gene|Endpoint",
                    "targets|associated_with",
                    "train",
                ),
                (
                    "P_test_1",
                    "D1",
                    "E_test",
                    "path_test_unique",
                    "D1|PW1|E_test",
                    "Drug|Pathway|Endpoint",
                    "participates_in|associated_with",
                    "test",
                ),
            ],
        )

        templates = run_template_extraction(config)
        written = read_templates(Path(config["output_dir"]) / "templates.tsv")

        expected_id = make_template_id("Drug|Gene|Endpoint", "targets|associated_with")
        assert [template.template_id for template in templates] == [expected_id]
        assert [template.template_id for template in written] == [expected_id]
        assert written[0].support_count == 2
        assert written[0].pair_ids == ("P_train_1", "P_train_2")
        assert written[0].path_ids == ("path_1", "path_2")

        test_only_id = make_template_id("Drug|Pathway|Endpoint", "participates_in|associated_with")
        assert test_only_id not in {template.template_id for template in written}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_min_support_filters_low_support_templates():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _config(tmp_dir, min_support=2)
        _write_pair_splits(Path(config["pair_split_dir"]))
        _write_gold_paths(
            Path(config["gold_path_file"]),
            [
                (
                    "P_train_1",
                    "D1",
                    "E_train",
                    "path_keep_1",
                    "D1|G1|E_train",
                    "Drug|Gene|Endpoint",
                    "targets|associated_with",
                    "train",
                ),
                (
                    "P_train_2",
                    "D2",
                    "E_train",
                    "path_keep_2",
                    "D2|G2|E_train",
                    "Drug|Gene|Endpoint",
                    "targets|associated_with",
                    "train",
                ),
                (
                    "P_valid_1",
                    "D1",
                    "E_valid",
                    "path_drop",
                    "D1|PW1|E_valid",
                    "Drug|Pathway|Endpoint",
                    "participates_in|associated_with",
                    "valid",
                ),
            ],
        )

        templates = run_template_extraction(config)

        assert len(templates) == 1
        assert templates[0].support_count == 2
        assert templates[0].node_type_sequence == "Drug|Gene|Endpoint"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "bad_row, match",
    [
        (
            ("", "D1", "E_train", "path_missing_pair", "D1|G1|E_train", "Drug|Gene|Endpoint", "targets|associated_with", "train"),
            "pair_id",
        ),
        (
            ("P_train_1", "D1", "E_train", "path_empty", "", "Drug|Gene|Endpoint", "targets|associated_with", "train"),
            "path_node_ids",
        ),
        (
            ("P_train_1", "D1", "E_train", "path_bad_nodes", "D1|G1|E_train", "Drug|Gene", "targets|associated_with", "train"),
            "node_type length",
        ),
        (
            ("P_train_1", "D1", "E_train", "path_bad_rel", "D1|G1|E_train", "Drug|Gene|Endpoint", "targets", "train"),
            "relation_type length",
        ),
    ],
)
def test_malformed_paths_raise_clear_errors(bad_row, match):
    tmp_dir = _workspace_tmp_dir()
    try:
        gold_path = tmp_dir / "gold_paths.tsv"
        _write_gold_paths(gold_path, [bad_row])
        with pytest.raises(ValueError, match=match):
            read_gold_paths(gold_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_gold_path_pair_id_must_exist_in_corresponding_split():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _config(tmp_dir)
        _write_pair_splits(Path(config["pair_split_dir"]))
        _write_gold_paths(
            Path(config["gold_path_file"]),
            [
                (
                    "P_missing",
                    "D1",
                    "E_train",
                    "path_missing",
                    "D1|G1|E_train",
                    "Drug|Gene|Endpoint",
                    "targets|associated_with",
                    "train",
                )
            ],
        )

        with pytest.raises(ValueError, match="not found in the corresponding pair split file"):
            run_template_extraction(config)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
