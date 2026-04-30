"""Evaluation for pair prediction plus supervised template prediction."""

from __future__ import annotations

from typing import Sequence

from mechrep.evaluation.eval_prediction import evaluate_predictions


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _macro_f1(true_ids: Sequence[str], pred_ids: Sequence[str]) -> float:
    labels = sorted(set(true_ids) | set(pred_ids))
    f1_values = []
    for label in labels:
        tp = sum(1 for true, pred in zip(true_ids, pred_ids) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(true_ids, pred_ids) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(true_ids, pred_ids) if true == label and pred != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def _top_ids(row: dict) -> list[str]:
    if "template_top3_ids" not in row or not row["template_top3_ids"]:
        return [row.get("predicted_template_id", "")]
    return [template_id for template_id in str(row["template_top3_ids"]).split("|") if template_id]


def evaluate_template_predictions(rows: Sequence[dict], *, num_templates: int | None = None) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")

    positive_rows = [row for row in rows if int(str(row["label"])) == 1]
    gold_rows = [
        row
        for row in positive_rows
        if _as_bool(row.get("has_gold_template", False)) and row.get("primary_template_id", "")
    ]
    coverage = len(gold_rows) / len(positive_rows) if positive_rows else 0.0

    if not gold_rows:
        return {
            "template_accuracy": None,
            "template_top3_accuracy": None,
            "template_macro_f1": None,
            "gold_template_coverage": coverage,
            "num_gold_template_examples": 0,
            "template_warning": "No gold-template-labeled positive examples are available for this split.",
        }

    true_ids = [row["primary_template_id"] for row in gold_rows]
    pred_ids = [row["predicted_template_id"] for row in gold_rows]
    correct = sum(1 for true, pred in zip(true_ids, pred_ids) if true == pred)
    metrics = {
        "template_accuracy": correct / len(gold_rows),
        "template_top3_accuracy": None,
        "template_macro_f1": _macro_f1(true_ids, pred_ids),
        "gold_template_coverage": coverage,
        "num_gold_template_examples": len(gold_rows),
        "template_warning": None,
    }

    if num_templates is not None and num_templates >= 3:
        if all(row.get("template_top3_ids") for row in gold_rows):
            top3_correct = sum(1 for row in gold_rows if row["primary_template_id"] in _top_ids(row)[:3])
            metrics["template_top3_accuracy"] = top3_correct / len(gold_rows)
        else:
            metrics["template_warning"] = "template_top3_accuracy unavailable because template_top3_ids are missing."
    return metrics


def evaluate_gold_template_predictions(
    rows: Sequence[dict],
    *,
    k_values: Sequence[int] = (10, 50, 100),
    group_by: str | None = None,
    num_templates: int | None = None,
) -> dict:
    metrics = evaluate_predictions(rows, k_values=k_values, group_by=group_by)
    metrics.update(evaluate_template_predictions(rows, num_templates=num_templates))
    return metrics
