"""Evaluation for semi-supervised template-guided training."""

from __future__ import annotations

from typing import Sequence

from mechrep.evaluation.eval_gold_template import _as_bool, _macro_f1, _top_ids
from mechrep.evaluation.eval_prediction import evaluate_predictions


def evaluate_semi_template_metrics(rows: Sequence[dict], *, num_templates: int | None = None) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")
    positive_rows = [row for row in rows if int(str(row["label"])) == 1]
    gold_rows = [
        row
        for row in positive_rows
        if _as_bool(row.get("has_gold_template", False)) and row.get("gold_template_id", "")
    ]
    no_gold_positive_rows = [row for row in positive_rows if not _as_bool(row.get("has_gold_template", False))]
    pseudo_rows = [
        row
        for row in no_gold_positive_rows
        if _as_bool(row.get("has_pseudo_template", False)) and row.get("pseudo_template_id", "")
    ]
    gold_coverage = len(gold_rows) / len(positive_rows) if positive_rows else 0.0
    pseudo_coverage = len(pseudo_rows) / len(no_gold_positive_rows) if no_gold_positive_rows else 0.0

    metrics = {
        "gold_template_coverage": gold_coverage,
        "pseudo_template_training_coverage": pseudo_coverage,
        "num_gold_template_examples": len(gold_rows),
        "num_pseudo_template_examples": len(pseudo_rows),
    }
    if not gold_rows:
        metrics.update(
            {
                "template_accuracy_on_gold": None,
                "template_top3_accuracy_on_gold": None,
                "template_macro_f1_on_gold": None,
                "template_warning": "No gold-template-labeled positive examples are available for this split.",
            }
        )
        return metrics

    true_ids = [row["gold_template_id"] for row in gold_rows]
    pred_ids = [row["predicted_template_id"] for row in gold_rows]
    metrics.update(
        {
            "template_accuracy_on_gold": sum(1 for true, pred in zip(true_ids, pred_ids) if true == pred)
            / len(gold_rows),
            "template_top3_accuracy_on_gold": None,
            "template_macro_f1_on_gold": _macro_f1(true_ids, pred_ids),
            "template_warning": None,
        }
    )
    if num_templates is not None and num_templates >= 3:
        if all(row.get("template_top3_ids") for row in gold_rows):
            metrics["template_top3_accuracy_on_gold"] = (
                sum(1 for row in gold_rows if row["gold_template_id"] in _top_ids(row)[:3]) / len(gold_rows)
            )
        else:
            metrics["template_warning"] = "template_top3_accuracy_on_gold unavailable because template_top3_ids are missing."
    return metrics


def evaluate_semi_template_predictions(
    rows: Sequence[dict],
    *,
    k_values: Sequence[int] = (10, 50, 100),
    group_by: str | None = None,
    num_templates: int | None = None,
) -> dict:
    metrics = evaluate_predictions(rows, k_values=k_values, group_by=group_by)
    metrics.update(evaluate_semi_template_metrics(rows, num_templates=num_templates))
    return metrics
