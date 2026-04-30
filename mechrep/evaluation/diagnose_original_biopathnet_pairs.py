"""Diagnostic report for original BioPathNet pairwise scores."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

from mechrep.data.build_pairs import PairRecord, read_pair_tsv
from mechrep.evaluation.eval_prediction import evaluate_predictions, write_metrics_json
from mechrep.evaluation.score_original_biopathnet_pairs import (
    _load_solver,
    _write_predictions,
    find_latest_checkpoint,
    infer_relation_name,
    load_yaml_config,
    score_records,
)
from mechrep.training.run_original_biopathnet_linux import prepare_environment, repository_root


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a quantile for an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def summarize_scores(rows: Sequence[dict]) -> dict:
    """Summarize score distributions overall and by binary label."""

    def summarize(values: Sequence[float]) -> dict:
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": ordered[0],
            "p01": _quantile(ordered, 0.01),
            "p10": _quantile(ordered, 0.10),
            "p25": _quantile(ordered, 0.25),
            "median": _quantile(ordered, 0.50),
            "p75": _quantile(ordered, 0.75),
            "p90": _quantile(ordered, 0.90),
            "p99": _quantile(ordered, 0.99),
            "max": ordered[-1],
            "mean": mean(ordered),
            "std": pstdev(ordered) if len(ordered) > 1 else 0.0,
        }

    positive = [float(row["score"]) for row in rows if int(row["label"]) == 1]
    negative = [float(row["score"]) for row in rows if int(row["label"]) == 0]
    if not positive or not negative:
        raise ValueError("Score diagnostics require both positive and negative rows")
    positive_summary = summarize(positive)
    negative_summary = summarize(negative)
    return {
        "all": summarize([float(row["score"]) for row in rows]),
        "positive": positive_summary,
        "negative": negative_summary,
        "mean_positive_minus_negative": positive_summary["mean"] - negative_summary["mean"],
        "median_positive_minus_negative": positive_summary["median"] - negative_summary["median"],
    }


def summarize_pair_records(records: Sequence[PairRecord]) -> dict:
    labels = [record.label for record in records]
    return {
        "num_pairs": len(records),
        "num_positive": sum(labels),
        "num_negative": len(labels) - sum(labels),
        "positive_fraction": sum(labels) / len(labels),
        "num_unique_drugs": len({record.drug_id for record in records}),
        "num_unique_endpoints": len({record.endpoint_id for record in records}),
        "num_unique_positive_endpoints": len({record.endpoint_id for record in records if record.label == 1}),
        "num_unique_negative_endpoints": len({record.endpoint_id for record in records if record.label == 0}),
    }


def limit_records(records: Sequence[PairRecord], *, max_records: int, seed: int) -> list[PairRecord]:
    if max_records <= 0 or len(records) <= max_records:
        return list(records)
    rng = random.Random(seed)
    positives = [record for record in records if record.label == 1]
    negatives = [record for record in records if record.label == 0]
    target_pos = min(len(positives), max_records // 2)
    target_neg = min(len(negatives), max_records - target_pos)
    if target_pos + target_neg < max_records:
        remaining_pos = len(positives) - target_pos
        remaining_neg = len(negatives) - target_neg
        take_more_pos = min(remaining_pos, max_records - target_pos - target_neg)
        target_pos += take_more_pos
        target_neg += min(remaining_neg, max_records - target_pos - target_neg)
    sampled = rng.sample(positives, target_pos) + rng.sample(negatives, target_neg)
    return sorted(sampled, key=lambda record: record.pair_id)


def endpoint_overlap_report(records_by_split: dict[str, Sequence[PairRecord]]) -> dict:
    endpoints = {split: {record.endpoint_id for record in records} for split, records in records_by_split.items()}
    drugs = {split: {record.drug_id for record in records} for split, records in records_by_split.items()}

    def overlap(kind: str, values: dict[str, set[str]]) -> dict:
        splits = sorted(values)
        pairwise = {}
        for i, left in enumerate(splits):
            for right in splits[i + 1 :]:
                shared = values[left] & values[right]
                pairwise[f"{left}_{right}"] = {
                    "count": len(shared),
                    "examples": sorted(shared)[:20],
                }
        return {
            "kind": kind,
            "unique_counts": {split: len(values[split]) for split in splits},
            "pairwise_overlap": pairwise,
        }

    return {
        "endpoints": overlap("endpoint_id", endpoints),
        "drugs": overlap("drug_id", drugs),
        "endpoint_ood_leakage_flags": {
            "train_valid_endpoint_overlap": len(endpoints.get("train", set()) & endpoints.get("valid", set())),
            "train_test_endpoint_overlap": len(endpoints.get("train", set()) & endpoints.get("test", set())),
            "valid_test_endpoint_overlap": len(endpoints.get("valid", set()) & endpoints.get("test", set())),
        },
    }


def relation_counts_for_files(dataset_dir: str | Path, files: Sequence[str]) -> dict:
    dataset_dir = Path(dataset_dir)
    report = {}
    for file_name in files:
        path = dataset_dir / file_name
        if not path.exists():
            report[file_name] = {"exists": False}
            continue
        counts = {}
        num_rows = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) != 3:
                    raise ValueError(f"{path} contains a non-triplet row: {row}")
                counts[row[1]] = counts.get(row[1], 0) + 1
                num_rows += 1
        report[file_name] = {
            "exists": True,
            "num_rows": num_rows,
            "num_relations": len(counts),
            "relation_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        }
    return report


def factgraph_support_report(
    dataset_dir: str | Path,
    records_by_split: dict[str, Sequence[PairRecord]],
    *,
    fact_file: str = "train1.txt",
) -> dict:
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / fact_file
    if not path.exists():
        return {"enabled": True, "exists": False, "fact_file": str(path)}

    split_drugs = {split: {record.drug_id for record in records} for split, records in records_by_split.items()}
    split_endpoints = {split: {record.endpoint_id for record in records} for split, records in records_by_split.items()}
    all_drugs = set().union(*split_drugs.values()) if split_drugs else set()
    all_endpoints = set().union(*split_endpoints.values()) if split_endpoints else set()
    drug_out_degree = {drug_id: 0 for drug_id in all_drugs}
    drug_incident_degree = {drug_id: 0 for drug_id in all_drugs}
    endpoint_in_degree = {endpoint_id: 0 for endpoint_id in all_endpoints}
    endpoint_incident_degree = {endpoint_id: 0 for endpoint_id in all_endpoints}
    num_rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for head, _relation, tail in reader:
            num_rows += 1
            if head in drug_out_degree:
                drug_out_degree[head] += 1
                drug_incident_degree[head] += 1
            if tail in drug_incident_degree:
                drug_incident_degree[tail] += 1
            if tail in endpoint_in_degree:
                endpoint_in_degree[tail] += 1
                endpoint_incident_degree[tail] += 1
            if head in endpoint_incident_degree:
                endpoint_incident_degree[head] += 1

    def split_summary(split: str) -> dict:
        drugs = split_drugs[split]
        endpoints = split_endpoints[split]
        drug_out = [drug_out_degree[drug] for drug in drugs]
        endpoint_in = [endpoint_in_degree[endpoint] for endpoint in endpoints]
        endpoint_incident = [endpoint_incident_degree[endpoint] for endpoint in endpoints]
        return {
            "num_drugs": len(drugs),
            "num_drugs_with_out_edges": sum(1 for degree in drug_out if degree > 0),
            "mean_drug_out_degree": mean(drug_out) if drug_out else 0.0,
            "num_endpoints": len(endpoints),
            "num_endpoints_with_in_edges": sum(1 for degree in endpoint_in if degree > 0),
            "mean_endpoint_in_degree": mean(endpoint_in) if endpoint_in else 0.0,
            "num_endpoints_with_any_incident_edge": sum(1 for degree in endpoint_incident if degree > 0),
            "mean_endpoint_incident_degree": mean(endpoint_incident) if endpoint_incident else 0.0,
        }

    return {
        "enabled": True,
        "exists": True,
        "fact_file": str(path),
        "num_fact_rows_scanned": num_rows,
        "splits": {split: split_summary(split) for split in sorted(records_by_split)},
    }


def run_diagnostic(
    *,
    config_path: str | Path,
    checkpoint: str | Path | None,
    checkpoint_dir: str | Path | None,
    split_dir: str | Path,
    output_dir: str | Path | None,
    relation_name: str | None,
    splits: Sequence[str],
    batch_size: int,
    k_values: Sequence[int],
    group_by: str | None,
    max_records_per_split: int,
    sampling_seed: int,
    include_factgraph_support: bool,
    progress_bar: bool,
) -> dict:
    root = repository_root()
    prepare_environment(root)
    config_path = Path(config_path).resolve(strict=False)
    config = load_yaml_config(config_path)
    dataset_dir = Path(config["dataset"]["path"])
    checkpoint_path = (
        Path(checkpoint).resolve(strict=False)
        if checkpoint
        else find_latest_checkpoint(checkpoint_dir or config["output_dir"]).resolve(strict=False)
    )
    output_path = Path(output_dir) if output_dir else checkpoint_path.parent / "pairwise_diagnostic"
    output_path.mkdir(parents=True, exist_ok=True)
    relation = relation_name or infer_relation_name(dataset_dir)

    _, dataset, solver = _load_solver(config_path, checkpoint_path)
    relation_id = dataset.inv_relation_vocab.get(relation)
    if relation_id is None:
        raise ValueError(f"Relation {relation!r} is missing from BioPathNet relation vocabulary")

    split_dir = Path(split_dir)
    available_records = {split: read_pair_tsv(split_dir / f"{split}.tsv", split=split) for split in splits}
    scored_records = {
        split: limit_records(records, max_records=max_records_per_split, seed=sampling_seed)
        for split, records in available_records.items()
    }

    report = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_path),
        "relation": {
            "name": relation,
            "id": int(relation_id),
            "inferred_from_train2": infer_relation_name(dataset_dir),
            "target_file_relation_counts": relation_counts_for_files(dataset_dir, ["train2.txt", "valid.txt", "test.txt"]),
        },
        "sampling": {
            "max_records_per_split": max_records_per_split,
            "sampling_seed": sampling_seed,
            "note": "0 means all records; positive values use deterministic label-balanced sampling per split.",
        },
        "split_input_summary": {
            split: {
                "available": summarize_pair_records(available_records[split]),
                "scored": summarize_pair_records(scored_records[split]),
            }
            for split in splits
        },
        "overlap": endpoint_overlap_report(available_records),
        "splits": {},
    }
    if include_factgraph_support:
        report["factgraph_support"] = factgraph_support_report(dataset_dir, available_records)

    for split in splits:
        rows = score_records(
            scored_records[split],
            dataset=dataset,
            solver=solver,
            relation_name=relation,
            split=split,
            batch_size=batch_size,
            progress_bar=progress_bar,
        )
        prediction_path = output_path / f"predictions_{split}.tsv"
        metrics_path = output_path / f"metrics_{split}.json"
        _write_predictions(prediction_path, rows)
        metrics = evaluate_predictions(rows, k_values=k_values, group_by=group_by)
        metrics["num_examples"] = len(rows)
        metrics["num_positive"] = sum(int(row["label"]) for row in rows)
        metrics["num_negative"] = len(rows) - metrics["num_positive"]
        metrics["num_available_examples"] = len(available_records[split])
        metrics["num_scored_examples"] = len(scored_records[split])
        score_summary = summarize_scores(rows)
        write_metrics_json(metrics_path, metrics)
        with (output_path / f"score_distribution_{split}.json").open("w", encoding="utf-8") as handle:
            json.dump(score_summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        report["splits"][split] = {
            "predictions": str(prediction_path),
            "metrics": str(metrics_path),
            "metrics_summary": metrics,
            "score_distribution": score_summary,
        }

    with (output_path / "diagnostic_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--relation-name")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--group-by", default="endpoint_id")
    parser.add_argument("--max-records-per-split", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--include-factgraph-support", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_by = args.group_by
    if group_by in ("", "none", "None"):
        group_by = None
    result = run_diagnostic(
        config_path=args.config,
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        relation_name=args.relation_name,
        splits=args.splits,
        batch_size=args.batch_size,
        k_values=args.k,
        group_by=group_by,
        max_records_per_split=args.max_records_per_split,
        sampling_seed=args.sampling_seed,
        include_factgraph_support=args.include_factgraph_support,
        progress_bar=not args.no_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
