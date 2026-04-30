"""Prediction metrics for endpoint-level drug repurposing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence


def _as_label(value: object) -> int:
    label = int(str(value))
    if label not in (0, 1):
        raise ValueError(f"Labels must be binary 0/1, got {value!r}")
    return label


def _validate_two_classes(labels: Sequence[int]) -> None:
    classes = set(labels)
    if classes != {0, 1}:
        raise ValueError(
            "AUROC and AUPRC require both positive and negative labels; "
            f"observed classes: {sorted(classes)}"
        )


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    _validate_two_classes(labels)
    ranks = _average_ranks(scores)
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (pos_rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)


def compute_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    _validate_two_classes(labels)
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    num_pos = sum(labels)
    seen_pos = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            seen_pos += 1
            precision_sum += seen_pos / rank
    return precision_sum / num_pos


def _group_rows(rows: Sequence[dict], group_by: str | None) -> List[List[dict]]:
    if group_by is None:
        return [list(rows)]
    groups = {}
    for row in rows:
        if group_by not in row:
            raise ValueError(f"group_by column {group_by!r} is missing from predictions")
        groups.setdefault(row[group_by], []).append(row)
    return [groups[key] for key in sorted(groups)]


def compute_topk_metrics(rows: Sequence[dict], k_values: Sequence[int], *, group_by: str | None = None) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")

    metrics = {}
    groups = _group_rows(rows, group_by)
    for k in k_values:
        if k <= 0:
            raise ValueError(f"K values must be positive, got {k}")
        recalls = []
        hits = []
        for group in groups:
            labels = [_as_label(row["label"]) for row in group]
            positives = sum(labels)
            if positives == 0:
                continue
            ranked = sorted(group, key=lambda row: (-float(row["score"]), row["pair_id"]))
            top_k = ranked[:k]
            found = sum(_as_label(row["label"]) for row in top_k)
            recalls.append(found / positives)
            hits.append(1.0 if found > 0 else 0.0)
        if not recalls:
            raise ValueError("Recall@K and Hits@K require at least one positive label")
        metrics[f"recall@{k}"] = sum(recalls) / len(recalls)
        metrics[f"hits@{k}"] = sum(hits) / len(hits)
    return metrics


def evaluate_predictions(
    rows: Sequence[dict],
    *,
    k_values: Sequence[int] = (10, 50, 100),
    group_by: str | None = None,
) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")
    labels = [_as_label(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    _validate_two_classes(labels)
    metrics = {
        "auroc": compute_auroc(labels, scores),
        "auprc": compute_average_precision(labels, scores),
    }
    metrics.update(compute_topk_metrics(rows, k_values, group_by=group_by))
    return metrics


def read_prediction_tsv(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"pair_id", "drug_id", "endpoint_id", "label", "score", "split"}
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def write_metrics_json(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate_prediction_file(
    predictions_path: str | Path,
    *,
    k_values: Sequence[int] = (10, 50, 100),
    group_by: str | None = None,
) -> dict:
    return evaluate_predictions(read_prediction_tsv(predictions_path), k_values=k_values, group_by=group_by)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--k", type=int, nargs="*", default=[10, 50, 100])
    parser.add_argument("--group-by", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_prediction_file(args.predictions, k_values=args.k, group_by=args.group_by)
    if args.output_json:
        write_metrics_json(args.output_json, metrics)
    else:
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
