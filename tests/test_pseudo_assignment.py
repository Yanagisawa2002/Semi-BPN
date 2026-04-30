import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.templates.pseudo_assign import run_pseudo_assignment
from mechrep.templates.template_vocab import TemplateVocab


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"pseudo_assignment_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def _read_tsv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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
PREDICTION_COLUMNS = [
    "pair_id",
    "drug_id",
    "endpoint_id",
    "label",
    "score",
    "split",
    "has_gold_template",
    "primary_template_id",
    "predicted_template_id",
    "template_confidence",
]
EVIDENCE_COLUMNS = [
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "path_id",
    "path_node_ids",
    "path_node_types",
    "path_relation_types",
    "path_score",
]


def _build_inputs(tmp_dir):
    split_dir = tmp_dir / "splits"
    template_dir = tmp_dir / "templates"
    pred_dir = tmp_dir / "predictions"
    evidence_dir = tmp_dir / "evidence"
    out_dir = tmp_dir / "out"
    template_id = "T_exact"
    other_template_id = "T_other"

    _write_tsv(
        split_dir / "train.tsv",
        PAIR_COLUMNS,
        [
            ("P_gold", "D1", "E_train", 1),
            ("P_assign", "D2", "E_train", 1),
            ("P_low_pair", "D3", "E_train", 1),
            ("P_no_path", "D4", "E_train", 1),
            ("P_bad_match", "D5", "E_train", 1),
            ("P_low_conf", "D6", "E_train", 1),
            ("P_negative", "D7", "E_train", 0),
        ],
    )
    _write_tsv(split_dir / "valid.tsv", PAIR_COLUMNS, [("P_valid", "D2", "E_valid", 1)])
    _write_tsv(split_dir / "test.tsv", PAIR_COLUMNS, [("P_test", "D2", "E_test", 1)])

    _write_tsv(
        template_dir / "templates.tsv",
        ["template_id", "node_type_sequence", "relation_type_sequence", "support_count", "pair_ids", "path_ids"],
        [
            (template_id, "Drug|Gene|Endpoint", "targets|associated_with", 5, "P_gold", "path_gold"),
            (other_template_id, "Drug|Pathway|Endpoint", "participates_in|associated_with", 1, "P_gold", "path_other"),
        ],
    )
    TemplateVocab.from_template_ids([template_id, other_template_id]).save(template_dir / "template_vocab.json")

    _write_tsv(
        template_dir / "gold_template_labels_train.tsv",
        LABEL_COLUMNS,
        [("P_gold", "D1", "E_train", "train", template_id, template_id, 1)],
    )
    _write_tsv(template_dir / "gold_template_labels_valid.tsv", LABEL_COLUMNS, [])
    _write_tsv(template_dir / "gold_template_labels_test.tsv", LABEL_COLUMNS, [])

    train_predictions = [
        ("P_gold", "D1", "E_train", 1, 0.99, "train", 1, template_id, template_id, 0.99),
        ("P_assign", "D2", "E_train", 1, 0.99, "train", 0, "", template_id, 0.99),
        ("P_low_pair", "D3", "E_train", 1, 0.20, "train", 0, "", template_id, 0.99),
        ("P_no_path", "D4", "E_train", 1, 0.99, "train", 0, "", template_id, 0.99),
        ("P_bad_match", "D5", "E_train", 1, 0.99, "train", 0, "", template_id, 0.99),
        ("P_low_conf", "D6", "E_train", 1, 0.80, "train", 0, "", template_id, 0.99),
        ("P_negative", "D7", "E_train", 0, 0.99, "train", 0, "", template_id, 0.99),
    ]
    _write_tsv(pred_dir / "predictions_train.tsv", PREDICTION_COLUMNS, train_predictions)
    _write_tsv(
        pred_dir / "predictions_valid.tsv",
        PREDICTION_COLUMNS,
        [("P_valid", "D2", "E_valid", 1, 0.99, "valid", 0, "", template_id, 0.99)],
    )
    _write_tsv(
        pred_dir / "predictions_test.tsv",
        PREDICTION_COLUMNS,
        [("P_test", "D2", "E_test", 1, 0.99, "test", 0, "", template_id, 0.99)],
    )

    _write_tsv(
        evidence_dir / "evidence_paths_train.tsv",
        EVIDENCE_COLUMNS,
        [
            ("P_gold", "D1", "E_train", "train", "path_gold", "D1|G|E_train", "Drug|Gene|Endpoint", "targets|associated_with", 1.0),
            ("P_assign", "D2", "E_train", "train", "path_assign", "D2|G|E_train", "Drug|Gene|Endpoint", "targets|associated_with", 1.0),
            ("P_low_pair", "D3", "E_train", "train", "path_low_pair", "D3|G|E_train", "Drug|Gene|Endpoint", "targets|associated_with", 1.0),
            ("P_bad_match", "D5", "E_train", "train", "path_bad", "D5|A|E_train", "Drug|Anatomy|Endpoint", "binds|causes", 1.0),
            ("P_low_conf", "D6", "E_train", "train", "path_low_conf", "D6|G|E_train", "Drug|Gene|Endpoint", "targets|associated_with", 1.0),
            ("P_negative", "D7", "E_train", "train", "path_negative", "D7|G|E_train", "Drug|Gene|Endpoint", "targets|associated_with", 1.0),
        ],
    )

    config = {
        "experiment": {"name": "test", "seed": 7, "output_dir": str(out_dir)},
        "data": {
            "split_dir": str(split_dir),
            "train_file": str(split_dir / "train.tsv"),
            "valid_file": str(split_dir / "valid.tsv"),
            "test_file": str(split_dir / "test.tsv"),
        },
        "templates": {
            "template_table": str(template_dir / "templates.tsv"),
            "template_vocab": str(template_dir / "template_vocab.json"),
            "gold_labels_train": str(template_dir / "gold_template_labels_train.tsv"),
            "gold_labels_valid": str(template_dir / "gold_template_labels_valid.tsv"),
            "gold_labels_test": str(template_dir / "gold_template_labels_test.tsv"),
        },
        "predictions": {
            "train_predictions": str(pred_dir / "predictions_train.tsv"),
            "valid_predictions": str(pred_dir / "predictions_valid.tsv"),
            "test_predictions": str(pred_dir / "predictions_test.tsv"),
        },
        "evidence_paths": {"train_paths": str(evidence_dir / "evidence_paths_train.tsv")},
        "matching": {"alpha_node": 0.4, "alpha_relation": 0.6},
        "pseudo_assignment": {
            "assignment_splits": ["train"],
            "tau_pair": 0.7,
            "tau_match": 0.8,
            "tau_confidence": 0.95,
            "w_pair": 0.4,
            "w_match": 0.4,
            "w_path": 0.2,
        },
    }
    return config


def test_pseudo_assignment_enforces_eligibility_thresholds_and_leakage_checks():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _build_inputs(tmp_dir)
        report = run_pseudo_assignment(config)
        out_dir = Path(config["experiment"]["output_dir"])
        assigned = _read_tsv(out_dir / "pseudo_template_labels_train.tsv")
        unassigned = {row["pair_id"]: row for row in _read_tsv(out_dir / "unassigned_no_gold_positives_train.tsv")}
        matches = _read_tsv(out_dir / "path_template_matches_train.tsv")

        assert [row["pair_id"] for row in assigned] == ["P_assign"]
        assert assigned[0]["pseudo_template_id"] == "T_exact"
        assert assigned[0]["matched_path_id"] == "path_assign"
        assert float(assigned[0]["final_confidence"]) >= 0.95

        assert unassigned["P_gold"]["reason_unassigned"] == "already_has_gold_template"
        assert unassigned["P_negative"]["reason_unassigned"] == "not_positive"
        assert unassigned["P_low_pair"]["reason_unassigned"] == "below_tau_pair"
        assert unassigned["P_no_path"]["reason_unassigned"] == "no_candidate_paths"
        assert unassigned["P_bad_match"]["reason_unassigned"] == "below_tau_match"
        assert unassigned["P_low_conf"]["reason_unassigned"] == "below_tau_confidence"

        assert len(matches) == 6
        assert report["assigned_no_gold_positive_pairs"] == 1
        assert report["leakage_checks"]["number_of_assigned_test_pairs"] == 0
        assert report["leakage_checks"]["number_of_assigned_valid_pairs"] == 0
        assert report["leakage_checks"]["number_of_assigned_negative_pairs"] == 0
        assert report["leakage_checks"]["number_of_assigned_gold_template_pairs"] == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_pseudo_assignment_outputs_are_deterministic_under_fixed_seed():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = _build_inputs(tmp_dir)
        first_report = run_pseudo_assignment(config)
        out_dir = Path(config["experiment"]["output_dir"])
        first_assignments = (out_dir / "pseudo_template_labels_train.tsv").read_text(encoding="utf-8")
        first_report_text = json.dumps(first_report, sort_keys=True)

        second_report = run_pseudo_assignment(config)
        second_assignments = (out_dir / "pseudo_template_labels_train.tsv").read_text(encoding="utf-8")
        second_report_text = json.dumps(second_report, sort_keys=True)

        assert second_assignments == first_assignments
        assert second_report_text == first_report_text
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
