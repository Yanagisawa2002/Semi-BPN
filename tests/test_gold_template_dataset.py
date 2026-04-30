import csv
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.gold_template_dataset import IGNORE_TEMPLATE_INDEX, GoldTemplatePairDataset
from mechrep.templates.template_vocab import TemplateVocab


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"gold_template_dataset_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


PAIR_COLUMNS = ["pair_id", "drug_id", "endpoint_id", "label"]
LABEL_COLUMNS = [
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "template_ids",
    "primary_template_id",
    "num_gold_paths",
]


def _write_splits(split_dir):
    _write_tsv(
        split_dir / "train.tsv",
        PAIR_COLUMNS,
        [
            ("P1", "D1", "E_train", "1"),
            ("P2", "D2", "E_train", "0"),
            ("P3", "D3", "E_train2", "1"),
        ],
    )
    _write_tsv(split_dir / "valid.tsv", PAIR_COLUMNS, [("PV", "D1", "E_valid", "1")])
    _write_tsv(split_dir / "test.tsv", PAIR_COLUMNS, [("PT", "D1", "E_test", "1")])


def _write_labels(tmp_dir):
    _write_tsv(
        tmp_dir / "labels_train.tsv",
        LABEL_COLUMNS,
        [
            ("P1", "D1", "E_train", "train", "T_a|T_b", "T_b", 2),
            ("P2", "D2", "E_train", "train", "T_a", "T_a", 1),
        ],
    )
    _write_tsv(tmp_dir / "labels_valid.tsv", LABEL_COLUMNS, [("PV", "D1", "E_valid", "valid", "T_a", "T_a", 1)])
    _write_tsv(tmp_dir / "labels_test.tsv", LABEL_COLUMNS, [("PT", "D1", "E_test", "test", "T_a", "T_a", 1)])


def test_gold_template_dataset_joins_labels_and_preserves_metadata():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir = tmp_dir / "splits"
        _write_splits(split_dir)
        _write_labels(tmp_dir)
        vocab = TemplateVocab.from_template_ids(["T_a", "T_b"])

        dataset = GoldTemplatePairDataset.from_files(
            split_dir=split_dir,
            train_labels=tmp_dir / "labels_train.tsv",
            valid_labels=tmp_dir / "labels_valid.tsv",
            test_labels=tmp_dir / "labels_test.tsv",
            template_vocab=vocab,
        )
        examples = {example.pair_id: example for example in dataset.examples("train")}

        assert examples["P1"].drug_id == "D1"
        assert examples["P1"].endpoint_id == "E_train"
        assert examples["P1"].label == 1
        assert examples["P1"].split == "train"
        assert examples["P1"].has_gold_template is True
        assert examples["P1"].template_ids == ("T_a", "T_b")
        assert examples["P1"].primary_template_index == vocab.index("T_b")

        assert examples["P3"].has_gold_template is False
        assert examples["P3"].primary_template_index == IGNORE_TEMPLATE_INDEX

        assert examples["P2"].label == 0
        assert examples["P2"].has_gold_template is False
        assert examples["P2"].primary_template_index == IGNORE_TEMPLATE_INDEX
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_gold_template_dataset_raises_for_unknown_template_by_default():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir = tmp_dir / "splits"
        _write_splits(split_dir)
        _write_labels(tmp_dir)
        vocab = TemplateVocab.from_template_ids(["T_a"])

        with pytest.raises(ValueError, match="unknown template IDs"):
            GoldTemplatePairDataset.from_files(
                split_dir=split_dir,
                train_labels=tmp_dir / "labels_train.tsv",
                valid_labels=tmp_dir / "labels_valid.tsv",
                test_labels=tmp_dir / "labels_test.tsv",
                template_vocab=vocab,
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_gold_template_dataset_can_ignore_unknown_templates_when_configured():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir = tmp_dir / "splits"
        _write_splits(split_dir)
        _write_labels(tmp_dir)
        vocab = TemplateVocab.from_template_ids(["T_a"])

        dataset = GoldTemplatePairDataset.from_files(
            split_dir=split_dir,
            train_labels=tmp_dir / "labels_train.tsv",
            valid_labels=tmp_dir / "labels_valid.tsv",
            test_labels=tmp_dir / "labels_test.tsv",
            template_vocab=vocab,
            unknown_template_policy="ignore",
        )

        p1 = {example.pair_id: example for example in dataset.examples("train")}["P1"]
        assert p1.primary_template_index == IGNORE_TEMPLATE_INDEX
        assert p1.has_gold_template is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
