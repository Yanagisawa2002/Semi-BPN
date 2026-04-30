import csv
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.gold_template_dataset import IGNORE_TEMPLATE_INDEX
from mechrep.data.semi_template_dataset import SemiTemplatePairDataset
from mechrep.templates.template_vocab import TemplateVocab


PAIR_COLUMNS = ["pair_id", "drug_id", "endpoint_id", "label"]
GOLD_COLUMNS = ["pair_id", "drug_id", "endpoint_id", "split", "template_ids", "primary_template_id", "num_gold_paths"]
PSEUDO_COLUMNS = [
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "pseudo_template_id",
    "pseudo_template_index",
    "matched_path_id",
    "pair_score",
    "template_match_score",
    "normalized_path_score",
    "final_confidence",
    "assignment_source",
]


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"semi_dataset_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _build_inputs(tmp_dir, *, overlap=False, test_pseudo=False):
    split_dir = tmp_dir / "splits"
    label_dir = tmp_dir / "labels"
    _write_tsv(
        split_dir / "train.tsv",
        PAIR_COLUMNS,
        [
            ("P_gold", "D1", "E_train", 1),
            ("P_pseudo", "D2", "E_train", 1),
            ("P_none", "D3", "E_train", 1),
            ("P_negative", "D4", "E_train", 0),
        ],
    )
    _write_tsv(split_dir / "valid.tsv", PAIR_COLUMNS, [("P_valid", "D1", "E_valid", 1)])
    _write_tsv(split_dir / "test.tsv", PAIR_COLUMNS, [("P_test", "D1", "E_test", 1)])
    _write_tsv(label_dir / "gold_train.tsv", GOLD_COLUMNS, [("P_gold", "D1", "E_train", "train", "T_gold", "T_gold", 1)])
    _write_tsv(label_dir / "gold_valid.tsv", GOLD_COLUMNS, [("P_valid", "D1", "E_valid", "valid", "T_gold", "T_gold", 1)])
    _write_tsv(label_dir / "gold_test.tsv", GOLD_COLUMNS, [("P_test", "D1", "E_test", "test", "T_gold", "T_gold", 1)])
    pseudo_rows = [
        ("P_pseudo", "D2", "E_train", "train", "T_pseudo", 1, "path_p", 0.9, 1.0, 1.0, 0.95, "high_confidence_path_template_match"),
        ("P_negative", "D4", "E_train", "train", "T_pseudo", 1, "path_n", 0.9, 1.0, 1.0, 0.95, "high_confidence_path_template_match"),
    ]
    if overlap:
        pseudo_rows.append(("P_gold", "D1", "E_train", "train", "T_pseudo", 1, "path_g", 0.9, 1.0, 1.0, 0.95, "high_confidence_path_template_match"))
    _write_tsv(label_dir / "pseudo_train.tsv", PSEUDO_COLUMNS, pseudo_rows)
    if test_pseudo:
        _write_tsv(
            label_dir / "pseudo_test.tsv",
            PSEUDO_COLUMNS,
            [("P_test", "D1", "E_test", "test", "T_pseudo", 1, "path_t", 0.9, 1.0, 1.0, 0.95, "high_confidence_path_template_match")],
        )
    return split_dir, label_dir


def _dataset(split_dir, label_dir, *, pseudo_by_split=None, use_pseudo_splits=("train",), allow_overlap=False, allow_test=False):
    return SemiTemplatePairDataset.from_files(
        split_dir=split_dir,
        gold_labels_train=label_dir / "gold_train.tsv",
        gold_labels_valid=label_dir / "gold_valid.tsv",
        gold_labels_test=label_dir / "gold_test.tsv",
        pseudo_labels_by_split=pseudo_by_split or {"train": label_dir / "pseudo_train.tsv"},
        template_vocab=TemplateVocab.from_template_ids(["T_gold", "T_pseudo"]),
        use_pseudo_splits=use_pseudo_splits,
        allow_gold_pseudo_overlap=allow_overlap,
        allow_test_pseudo=allow_test,
    )


def test_semi_dataset_joins_gold_and_pseudo_and_preserves_metadata():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir, label_dir = _build_inputs(tmp_dir)
        dataset = _dataset(split_dir, label_dir)
        examples = {example.pair_id: example for example in dataset.examples("train")}

        assert examples["P_gold"].has_gold_template is True
        assert examples["P_gold"].gold_template_id == "T_gold"
        assert examples["P_gold"].template_supervision_source == "gold"
        assert examples["P_gold"].pseudo_template_index == IGNORE_TEMPLATE_INDEX

        assert examples["P_pseudo"].has_pseudo_template is True
        assert examples["P_pseudo"].pseudo_template_id == "T_pseudo"
        assert examples["P_pseudo"].template_supervision_source == "pseudo"

        assert examples["P_negative"].label == 0
        assert examples["P_negative"].has_pseudo_template is False
        assert examples["P_negative"].template_supervision_source == "none"
        assert examples["P_none"].gold_template_index == IGNORE_TEMPLATE_INDEX
        assert examples["P_none"].pseudo_template_index == IGNORE_TEMPLATE_INDEX
        assert examples["P_pseudo"].drug_id == "D2"
        assert examples["P_pseudo"].endpoint_id == "E_train"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_gold_pseudo_overlap_raises_by_default_and_can_be_ignored():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir, label_dir = _build_inputs(tmp_dir, overlap=True)
        with pytest.raises(ValueError, match="both gold-template and pseudo-template"):
            _dataset(split_dir, label_dir)

        dataset = _dataset(split_dir, label_dir, allow_overlap=True)
        gold = {example.pair_id: example for example in dataset.examples("train")}["P_gold"]
        assert gold.template_supervision_source == "gold"
        assert gold.has_pseudo_template is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_test_pseudo_labels_are_rejected_by_default():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir, label_dir = _build_inputs(tmp_dir, test_pseudo=True)
        with pytest.raises(ValueError, match="Test pseudo-template labels are rejected"):
            _dataset(
                split_dir,
                label_dir,
                pseudo_by_split={"test": label_dir / "pseudo_test.tsv"},
                use_pseudo_splits=("test",),
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_pseudo_template_cap_keeps_highest_confidence_per_template_without_dropping_pairs():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir = tmp_dir / "splits"
        label_dir = tmp_dir / "labels"
        _write_tsv(
            split_dir / "train.tsv",
            PAIR_COLUMNS,
            [
                ("P_a", "D1", "E_train", 1),
                ("P_b", "D2", "E_train", 1),
                ("P_c", "D3", "E_train", 1),
                ("P_other", "D4", "E_train", 1),
            ],
        )
        _write_tsv(split_dir / "valid.tsv", PAIR_COLUMNS, [("P_valid", "D1", "E_valid", 1)])
        _write_tsv(split_dir / "test.tsv", PAIR_COLUMNS, [("P_test", "D1", "E_test", 1)])
        _write_tsv(label_dir / "gold_train.tsv", GOLD_COLUMNS, [])
        _write_tsv(label_dir / "gold_valid.tsv", GOLD_COLUMNS, [("P_valid", "D1", "E_valid", "valid", "T_gold", "T_gold", 1)])
        _write_tsv(label_dir / "gold_test.tsv", GOLD_COLUMNS, [("P_test", "D1", "E_test", "test", "T_gold", "T_gold", 1)])
        _write_tsv(
            label_dir / "pseudo_train.tsv",
            PSEUDO_COLUMNS,
            [
                ("P_a", "D1", "E_train", "train", "T_pseudo", 1, "path_a", 0.9, 1.0, 1.0, 0.80, "high_confidence_path_template_match"),
                ("P_b", "D2", "E_train", "train", "T_pseudo", 1, "path_b", 0.9, 1.0, 1.0, 0.95, "high_confidence_path_template_match"),
                ("P_c", "D3", "E_train", "train", "T_pseudo", 1, "path_c", 0.9, 1.0, 1.0, 0.90, "high_confidence_path_template_match"),
                ("P_other", "D4", "E_train", "train", "T_other", 2, "path_o", 0.9, 1.0, 1.0, 0.70, "high_confidence_path_template_match"),
            ],
        )

        dataset = SemiTemplatePairDataset.from_files(
            split_dir=split_dir,
            gold_labels_train=label_dir / "gold_train.tsv",
            gold_labels_valid=label_dir / "gold_valid.tsv",
            gold_labels_test=label_dir / "gold_test.tsv",
            pseudo_labels_by_split={"train": label_dir / "pseudo_train.tsv"},
            template_vocab=TemplateVocab.from_template_ids(["T_gold", "T_pseudo", "T_other"]),
            use_pseudo_splits=("train",),
            max_pseudo_per_template=2,
        )
        examples = {example.pair_id: example for example in dataset.examples("train")}

        assert len(examples) == 4
        assert examples["P_b"].has_pseudo_template is True
        assert examples["P_c"].has_pseudo_template is True
        assert examples["P_a"].has_pseudo_template is False
        assert examples["P_a"].template_supervision_source == "none"
        assert examples["P_other"].has_pseudo_template is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
