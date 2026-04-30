import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.evaluation.eval_gold_template import evaluate_gold_template_predictions, evaluate_template_predictions


def _rows_with_gold():
    return [
        {
            "pair_id": "P1",
            "drug_id": "D1",
            "endpoint_id": "E1",
            "label": "1",
            "score": "0.9",
            "split": "test",
            "has_gold_template": "1",
            "primary_template_id": "T_a",
            "predicted_template_id": "T_a",
            "template_confidence": "0.8",
            "template_top3_ids": "T_a|T_b|T_c",
        },
        {
            "pair_id": "P2",
            "drug_id": "D2",
            "endpoint_id": "E1",
            "label": "0",
            "score": "0.1",
            "split": "test",
            "has_gold_template": "0",
            "primary_template_id": "",
            "predicted_template_id": "T_b",
            "template_confidence": "0.7",
            "template_top3_ids": "T_b|T_a|T_c",
        },
        {
            "pair_id": "P3",
            "drug_id": "D3",
            "endpoint_id": "E2",
            "label": "1",
            "score": "0.2",
            "split": "test",
            "has_gold_template": "1",
            "primary_template_id": "T_b",
            "predicted_template_id": "T_c",
            "template_confidence": "0.6",
            "template_top3_ids": "T_c|T_b|T_a",
        },
        {
            "pair_id": "P4",
            "drug_id": "D4",
            "endpoint_id": "E2",
            "label": "0",
            "score": "0.3",
            "split": "test",
            "has_gold_template": "0",
            "primary_template_id": "",
            "predicted_template_id": "T_a",
            "template_confidence": "0.4",
            "template_top3_ids": "T_a|T_b|T_c",
        },
    ]


def test_template_metrics_are_computed_only_on_gold_template_examples():
    metrics = evaluate_gold_template_predictions(_rows_with_gold(), k_values=[1], group_by=None, num_templates=3)

    assert metrics["auroc"] == 0.75
    assert metrics["template_accuracy"] == 0.5
    assert metrics["template_top3_accuracy"] == 1.0
    assert metrics["gold_template_coverage"] == 1.0
    assert metrics["num_gold_template_examples"] == 2


def test_empty_gold_template_subset_reports_null_metrics_with_warning():
    rows = [
        {
            "pair_id": "P1",
            "drug_id": "D1",
            "endpoint_id": "E1",
            "label": "1",
            "score": "0.8",
            "split": "test",
            "has_gold_template": "0",
            "primary_template_id": "",
            "predicted_template_id": "T_a",
            "template_confidence": "0.5",
        },
        {
            "pair_id": "P2",
            "drug_id": "D2",
            "endpoint_id": "E1",
            "label": "0",
            "score": "0.2",
            "split": "test",
            "has_gold_template": "0",
            "primary_template_id": "",
            "predicted_template_id": "T_b",
            "template_confidence": "0.5",
        },
    ]

    metrics = evaluate_gold_template_predictions(rows, k_values=[1], group_by=None, num_templates=2)

    assert metrics["auprc"] == 1.0
    assert metrics["template_accuracy"] is None
    assert metrics["template_macro_f1"] is None
    assert metrics["gold_template_coverage"] == 0.0
    assert "No gold-template-labeled" in metrics["template_warning"]


def test_template_metrics_do_not_use_negative_examples_with_template_fields():
    rows = _rows_with_gold()
    rows[1]["has_gold_template"] = "1"
    rows[1]["primary_template_id"] = "T_b"
    rows[1]["predicted_template_id"] = "T_b"

    metrics = evaluate_template_predictions(rows, num_templates=3)

    assert metrics["num_gold_template_examples"] == 2
    assert metrics["template_accuracy"] == 0.5
