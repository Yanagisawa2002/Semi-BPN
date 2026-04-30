import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.evaluation.eval_prediction import evaluate_predictions


def test_prediction_evaluator_computes_toy_metrics():
    rows = [
        {"pair_id": "p0", "drug_id": "d0", "endpoint_id": "e0", "label": "1", "score": "0.9", "split": "test"},
        {"pair_id": "p1", "drug_id": "d1", "endpoint_id": "e0", "label": "0", "score": "0.8", "split": "test"},
        {"pair_id": "p2", "drug_id": "d2", "endpoint_id": "e0", "label": "0", "score": "0.2", "split": "test"},
        {"pair_id": "p3", "drug_id": "d3", "endpoint_id": "e0", "label": "1", "score": "0.1", "split": "test"},
    ]

    metrics = evaluate_predictions(rows, k_values=[1, 2], group_by=None)

    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["auprc"] == pytest.approx(0.75)
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["hits@1"] == pytest.approx(1.0)
    assert metrics["recall@2"] == pytest.approx(0.5)
    assert metrics["hits@2"] == pytest.approx(1.0)
    assert metrics["candidate_set"]["group_by"] == "__all__"
    assert metrics["candidate_set"]["num_groups"] == 1
    assert metrics["topk_warning"] is None


def test_prediction_evaluator_reports_group_candidate_saturation():
    rows = [
        {"pair_id": "p0", "drug_id": "d0", "endpoint_id": "e0", "label": "1", "score": "0.9", "split": "test"},
        {"pair_id": "p1", "drug_id": "d1", "endpoint_id": "e0", "label": "0", "score": "0.2", "split": "test"},
        {"pair_id": "p2", "drug_id": "d2", "endpoint_id": "e1", "label": "1", "score": "0.8", "split": "test"},
        {"pair_id": "p3", "drug_id": "d3", "endpoint_id": "e1", "label": "0", "score": "0.1", "split": "test"},
    ]

    metrics = evaluate_predictions(rows, k_values=[1, 5], group_by="endpoint_id")

    assert metrics["candidate_set"]["group_by"] == "endpoint_id"
    assert metrics["candidate_set"]["median_candidates_per_group"] == pytest.approx(2.0)
    assert metrics["candidate_set"]["fraction_groups_with_candidates_lte_5"] == pytest.approx(1.0)
    assert metrics["topk_warning"] is not None


def test_prediction_evaluator_raises_on_one_class_labels():
    rows = [
        {"pair_id": "p0", "drug_id": "d0", "endpoint_id": "e0", "label": "1", "score": "0.9", "split": "test"},
        {"pair_id": "p1", "drug_id": "d1", "endpoint_id": "e0", "label": "1", "score": "0.8", "split": "test"},
    ]

    with pytest.raises(ValueError, match="require both positive and negative labels"):
        evaluate_predictions(rows, k_values=[1], group_by=None)
