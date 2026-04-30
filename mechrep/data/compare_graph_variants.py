"""Compare BioPathNet graph variants used for smoke / profiling runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import yaml

from mechrep.data.build_path_subgraph import compute_gold_path_coverage, read_triples


COMPARISON_COLUMNS = (
    "variant",
    "train1_edges",
    "train1_nodes",
    "train1_size_mb",
    "edge_reduction_vs_full",
    "node_reduction_vs_full",
    "train_gold_path_coverage",
    "valid_gold_path_coverage",
    "test_gold_path_coverage",
    "pairs_with_evidence_paths",
    "fallback_pairs_considered",
    "fallback_pairs_with_any_structural_edge",
    "fallback_unique_edges_added",
    "smoke_command",
)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


def graph_stats(graph_dir: str | Path, *, gold_paths_file: str | Path | None) -> dict:
    graph_dir = Path(graph_dir)
    train1 = graph_dir / "train1.txt"
    if not train1.exists():
        raise FileNotFoundError(f"Missing graph file: {train1}")
    edge_set = set(read_triples(train1))
    nodes = {node for head, _, tail in edge_set for node in (head, tail)}
    return {
        "train1_edges": len(edge_set),
        "train1_nodes": len(nodes),
        "train1_size_mb": train1.stat().st_size / (1024 * 1024),
        "gold_path_coverage": compute_gold_path_coverage(gold_paths_file, edge_set),
    }


def load_variant_report(path: str | Path | None) -> dict:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {"missing_report": str(path)}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _coverage(report: dict, split: str) -> float | None:
    split_report = report.get("gold_path_coverage", {}).get(split, {})
    return split_report.get("gold_path_coverage")


def compare_graph_variants(config: dict) -> dict:
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Config must contain a non-empty variants list")
    output_config = config["output"]
    gold_paths_file = config.get("gold_paths_file")

    rows = []
    detailed = {}
    full_edges = None
    full_nodes = None
    for item in variants:
        if not isinstance(item, dict):
            raise ValueError("Each variant entry must be a mapping")
        name = item["name"]
        stats = graph_stats(item["graph_dir"], gold_paths_file=gold_paths_file)
        variant_report = load_variant_report(item.get("report"))
        if full_edges is None:
            full_edges = stats["train1_edges"]
            full_nodes = stats["train1_nodes"]
        fallback_stats = variant_report.get("fallback_structural_support", {}).get("stats", {})
        row = {
            "variant": name,
            "train1_edges": stats["train1_edges"],
            "train1_nodes": stats["train1_nodes"],
            "train1_size_mb": stats["train1_size_mb"],
            "edge_reduction_vs_full": full_edges / stats["train1_edges"] if stats["train1_edges"] else None,
            "node_reduction_vs_full": full_nodes / stats["train1_nodes"] if stats["train1_nodes"] else None,
            "train_gold_path_coverage": _coverage(stats, "train"),
            "valid_gold_path_coverage": _coverage(stats, "valid"),
            "test_gold_path_coverage": _coverage(stats, "test"),
            "pairs_with_evidence_paths": variant_report.get("pairs_with_evidence_paths"),
            "fallback_pairs_considered": fallback_stats.get("fallback_pairs_considered"),
            "fallback_pairs_with_any_structural_edge": fallback_stats.get("fallback_pairs_with_any_structural_edge"),
            "fallback_unique_edges_added": fallback_stats.get("fallback_unique_edges_added"),
            "smoke_command": item.get("smoke_command", ""),
        }
        rows.append(row)
        detailed[name] = {
            "graph_stats": stats,
            "variant_report": variant_report,
            "comparison_row": row,
        }

    output_tsv = Path(output_config["summary_tsv"])
    output_json = Path(output_config["summary_json"])
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "variants": rows,
        "details": detailed,
        "output_tsv": str(output_tsv),
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graph_variant_comparison.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_graph_variants(load_config(args.config))
    print(json.dumps(report["variants"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
