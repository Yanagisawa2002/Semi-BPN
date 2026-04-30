import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.evaluation.eval_semi_template import evaluate_semi_template_predictions


def test_semi_eval_uses_gold_labels_for_template_metrics_not_pseudo_labels():
    rows = [
        {
            "pair_id": "P_gold",
            "drug_id": "D1",
            "endpoint_id": "E1",
            "label": "1",
            "score": "0.9",
            "split": "valid",
            "has_gold_template": "1",
            "gold_template_id": "T_gold",
            "has_pseudo_template": "0",
            "pseudo_template_id": "",
            "template_supervision_source": "gold",
            "predicted_template_id": "T_gold",
            "template_confidence": "0.8",
            "template_top3_ids": "T_gold|T_pseudo|T_other",
        },
        {
            "pair_id": "P_pseudo",
            "drug_id": "D2",
            "endpoint_id": "E1",
            "label": "1",
            "score": "0.7",
            "split": "valid",
            "has_gold_template": "0",
            "gold_template_id": "",
            "has_pseudo_template": "1",
            "pseudo_template_id": "T_pseudo",
            "template_supervision_source": "pseudo",
            "predicted_template_id": "T_pseudo",
            "template_confidence": "0.9",
            "template_top3_ids": "T_pseudo|T_gold|T_other",
        },
        {
            "pair_id": "P_neg",
            "drug_id": "D3",
            "endpoint_id": "E1",
            "label": "0",
            "score": "0.1",
            "split": "valid",
            "has_gold_template": "0",
            "gold_template_id": "",
            "has_pseudo_template": "0",
            "pseudo_template_id": "",
            "template_supervision_source": "none",
            "predicted_template_id": "T_pseudo",
            "template_confidence": "0.5",
            "template_top3_ids": "T_pseudo|T_gold|T_other",
        },
    ]

    metrics = evaluate_semi_template_predictions(rows, k_values=[1], group_by=None, num_templates=3)

    assert metrics["auprc"] > 0
    assert metrics["template_accuracy_on_gold"] == 1.0
    assert metrics["template_top3_accuracy_on_gold"] == 1.0
    assert metrics["gold_template_coverage"] == 0.5
    assert metrics["pseudo_template_training_coverage"] == 1.0
    assert metrics["num_gold_template_examples"] == 1
    assert metrics["num_pseudo_template_examples"] == 1


def test_empty_gold_subset_reports_null_template_metrics_with_warning():
    rows = [
        {
            "pair_id": "P1",
            "drug_id": "D1",
            "endpoint_id": "E1",
            "label": "1",
            "score": "0.9",
            "split": "test",
            "has_gold_template": "0",
            "gold_template_id": "",
            "has_pseudo_template": "1",
            "pseudo_template_id": "T_pseudo",
            "template_supervision_source": "pseudo",
            "predicted_template_id": "T_pseudo",
            "template_confidence": "0.8",
        },
        {
            "pair_id": "P2",
            "drug_id": "D2",
            "endpoint_id": "E1",
            "label": "0",
            "score": "0.1",
            "split": "test",
            "has_gold_template": "0",
            "gold_template_id": "",
            "has_pseudo_template": "0",
            "pseudo_template_id": "",
            "template_supervision_source": "none",
            "predicted_template_id": "T_gold",
            "template_confidence": "0.8",
        },
    ]

    metrics = evaluate_semi_template_predictions(rows, k_values=[1], group_by=None, num_templates=2)

    assert metrics["template_accuracy_on_gold"] is None
    assert metrics["template_macro_f1_on_gold"] is None
    assert "No gold-template-labeled" in metrics["template_warning"]
    assert metrics["auroc"] == 1.0
