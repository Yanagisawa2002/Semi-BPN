"""Endpoint-OOD evaluation wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from mechrep.evaluation.eval_prediction import evaluate_prediction_file


def evaluate_ood_predictions(
    predictions_path: str | Path,
    *,
    k_values: Sequence[int] = (10, 50, 100),
    group_by: str | None = "endpoint_id",
) -> dict:
    return evaluate_prediction_file(predictions_path, k_values=k_values, group_by=group_by)
