import csv
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.templates.template_vocab import build_template_vocab


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"template_vocab_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


TEMPLATE_COLUMNS = ["template_id", "node_type_sequence", "relation_type_sequence", "support_count", "pair_ids", "path_ids"]
LABEL_COLUMNS = [
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "template_ids",
    "primary_template_id",
    "num_gold_paths",
]


def test_template_vocab_is_deterministic_and_excludes_test_only_templates():
    tmp_dir = _workspace_tmp_dir()
    try:
        template_table = tmp_dir / "templates.tsv"
        train_labels = tmp_dir / "gold_template_labels_train.tsv"
        _write_tsv(
            template_table,
            TEMPLATE_COLUMNS,
            [
                ("T_test_only", "Drug|Pathway|Endpoint", "x|y", 1, "P_test", "path_test"),
                ("T_b", "Drug|Gene|Endpoint", "b|c", 1, "P_train_2", "path_b"),
                ("T_a", "Drug|Gene|Endpoint", "a|c", 1, "P_train_1", "path_a"),
            ],
        )
        _write_tsv(
            train_labels,
            LABEL_COLUMNS,
            [
                ("P_train_2", "D2", "E1", "train", "T_b", "T_b", 1),
                ("P_train_1", "D1", "E1", "train", "T_a", "T_a", 1),
            ],
        )

        vocab = build_template_vocab(template_table, train_labels=train_labels, use_train_templates_only=True)

        assert vocab.template_id_to_index == {"T_a": 0, "T_b": 1}
        assert vocab.template_index_to_id == {0: "T_a", 1: "T_b"}
        assert "T_test_only" not in vocab.template_id_to_index
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_template_vocab_raises_for_unknown_training_template():
    tmp_dir = _workspace_tmp_dir()
    try:
        template_table = tmp_dir / "templates.tsv"
        train_labels = tmp_dir / "gold_template_labels_train.tsv"
        _write_tsv(template_table, TEMPLATE_COLUMNS, [("T_a", "Drug|Gene|Endpoint", "a|b", 1, "P1", "path1")])
        _write_tsv(train_labels, LABEL_COLUMNS, [("P1", "D1", "E1", "train", "T_missing", "T_missing", 1)])

        with pytest.raises(ValueError, match="not present in template table"):
            build_template_vocab(template_table, train_labels=train_labels, use_train_templates_only=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
