"""Diagnose train positive pairs that have no retrieved evidence paths."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from mechrep.data.build_pairs import PairRecord, read_pair_tsv
from mechrep.templates.export_evidence_paths import (
    DEFAULT_GUIDED_SEED_RELATIONS,
    DEFAULT_GUIDED_SEED_TYPES,
    NODE_TYPE_PRIORITY,
    RELATION_PRIORITY,
    load_entity_types,
)


NO_EVIDENCE_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "label",
    "has_gold_template",
    "evidence_path_count",
    "no_evidence_scope",
    "drug_in_entity_table",
    "endpoint_in_entity_table",
    "drug_out_degree",
    "drug_in_degree",
    "endpoint_in_degree",
    "endpoint_out_degree",
    "guided_drug_out_degree",
    "two_hop_bridge_count",
    "three_hop_bridge_count",
    "fallback_structural_edge_count",
    "sample_drug_out_edges",
    "sample_endpoint_in_edges",
    "sample_two_hop_bridge_nodes",
    "reason_unretrieved",
)


@dataclass(frozen=True)
class Edge:
    head: str
    relation: str
    tail: str

    def as_triple(self) -> tuple[str, str, str]:
        return self.head, self.relation, self.tail


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


def read_gold_pair_ids(path: str | Path) -> set[str]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        if "pair_id" not in reader.fieldnames:
            raise ValueError(f"{path} is missing required column pair_id")
        return {row["pair_id"] for row in reader if row.get("pair_id")}


def read_evidence_pair_counts(path: str | Path) -> Counter:
    path = Path(path)
    counts = Counter()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        for column in ("pair_id", "split"):
            if column not in reader.fieldnames:
                raise ValueError(f"{path} is missing required column {column}")
        non_train_rows = 0
        for row in reader:
            if row["split"] != "train":
                non_train_rows += 1
                continue
            counts[row["pair_id"]] += 1
    counts["_non_train_rows"] = non_train_rows
    return counts


def read_entity_set(entity_types_path: str | Path) -> set[str]:
    entities = set()
    with Path(entity_types_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"{entity_types_path}:{row_number} must have exactly two columns")
            entities.add(row[0])
    if not entities:
        raise ValueError(f"{entity_types_path} contains no entities")
    return entities


def read_train_kg_edges(path: str | Path) -> Iterable[Edge]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(f"{path}:{row_number} must have exactly three columns")
            yield Edge(row[0], row[1], row[2])


def _edge_priority(edge: Edge, *, node: str, entity_types: dict[str, str]) -> tuple:
    node_type = entity_types.get(node, "")
    return (
        RELATION_PRIORITY.get(edge.relation, 100),
        NODE_TYPE_PRIORITY.get(node_type, 100),
        node_type,
        node,
        edge.relation,
    )


def sort_out_edges(edges: Sequence[Edge], entity_types: dict[str, str]) -> list[Edge]:
    return sorted(edges, key=lambda edge: _edge_priority(edge, node=edge.tail, entity_types=entity_types))


def sort_in_edges(edges: Sequence[Edge], entity_types: dict[str, str]) -> list[Edge]:
    return sorted(edges, key=lambda edge: _edge_priority(edge, node=edge.head, entity_types=entity_types))


def build_structural_index(
    kg_path: str | Path,
    pairs: Sequence[PairRecord],
    *,
    entity_types: dict[str, str],
) -> dict:
    drugs = {pair.drug_id for pair in pairs}
    endpoints = {pair.endpoint_id for pair in pairs}
    relevant_nodes = drugs | endpoints
    out_degree = Counter()
    in_degree = Counter()
    drug_out = defaultdict(list)
    endpoint_in = defaultdict(list)
    guided_drug_out = defaultdict(list)

    for edge in read_train_kg_edges(kg_path):
        if edge.head in relevant_nodes:
            out_degree[edge.head] += 1
        if edge.tail in relevant_nodes:
            in_degree[edge.tail] += 1
        if edge.head in drugs:
            drug_out[edge.head].append(edge)
            tail_type = entity_types.get(edge.tail)
            if edge.relation in DEFAULT_GUIDED_SEED_RELATIONS and tail_type in DEFAULT_GUIDED_SEED_TYPES:
                guided_drug_out[edge.head].append(edge)
        if edge.tail in endpoints:
            endpoint_in[edge.tail].append(edge)

    first_hop_nodes = {edge.tail for edges in drug_out.values() for edge in edges}
    endpoint_predecessors = {edge.head for edges in endpoint_in.values() for edge in edges}
    mid_edges_by_head = defaultdict(list)
    if first_hop_nodes and endpoint_predecessors:
        for edge in read_train_kg_edges(kg_path):
            if edge.head in first_hop_nodes and edge.tail in endpoint_predecessors:
                mid_edges_by_head[edge.head].append(edge)

    return {
        "out_degree": out_degree,
        "in_degree": in_degree,
        "drug_out": {node: sort_out_edges(edges, entity_types) for node, edges in drug_out.items()},
        "endpoint_in": {node: sort_in_edges(edges, entity_types) for node, edges in endpoint_in.items()},
        "guided_drug_out": {
            node: sort_out_edges(edges, entity_types) for node, edges in guided_drug_out.items()
        },
        "mid_edges_by_head": {
            node: sort_out_edges(edges, entity_types) for node, edges in mid_edges_by_head.items()
        },
    }


def _format_edges(edges: Sequence[Edge], limit: int) -> str:
    return "|".join(f"{edge.head},{edge.relation},{edge.tail}" for edge in edges[:limit])


def _count_three_hop_bridges(
    drug_out_edges: Sequence[Edge],
    endpoint_in_edges: Sequence[Edge],
    mid_edges_by_head: dict[str, list[Edge]],
    *,
    cap: int = 100000,
) -> int:
    endpoint_predecessors = {edge.head for edge in endpoint_in_edges}
    count = 0
    for first_edge in drug_out_edges:
        for mid_edge in mid_edges_by_head.get(first_edge.tail, []):
            if mid_edge.tail in endpoint_predecessors:
                count += 1
                if count >= cap:
                    return count
    return count


def classify_reason(
    *,
    has_gold_template: bool,
    drug_in_entity_table: bool,
    endpoint_in_entity_table: bool,
    drug_out_degree: int,
    endpoint_in_degree: int,
    guided_drug_out_degree: int,
    two_hop_bridge_count: int,
    three_hop_bridge_count: int,
) -> str:
    if has_gold_template:
        return "gold_template_pair_excluded_from_retrieval"
    if not drug_in_entity_table:
        return "drug_missing_from_entity_table"
    if not endpoint_in_entity_table:
        return "endpoint_missing_from_entity_table"
    if drug_out_degree == 0:
        return "drug_has_no_out_edges"
    if endpoint_in_degree == 0:
        return "endpoint_has_no_in_edges"
    if guided_drug_out_degree == 0:
        return "drug_has_no_guided_seed_edges"
    if two_hop_bridge_count == 0 and three_hop_bridge_count == 0:
        return "no_two_or_three_hop_bridge"
    return "bridge_exists_but_retrieval_filter_or_depth_limit"


def build_no_evidence_report(config: dict) -> dict:
    data_config = config["data"]
    output_config = config["output"]
    analysis_config = config.get("analysis", {})
    sample_limit = int(analysis_config.get("max_sample_items", 5))

    train_pairs = read_pair_tsv(data_config["train_pairs"], split="train")
    train_positive_pairs = [pair for pair in train_pairs if pair.label == 1]
    gold_pair_ids = read_gold_pair_ids(data_config["gold_labels_train"])
    evidence_counts = read_evidence_pair_counts(data_config["evidence_paths_train"])
    evidence_pair_ids = {pair_id for pair_id, count in evidence_counts.items() if pair_id != "_non_train_rows" and count}
    no_evidence_pairs = [pair for pair in train_positive_pairs if pair.pair_id not in evidence_pair_ids]

    entity_types = load_entity_types(data_config["entity_types"], data_config["conversion_report"])
    entity_set = set(entity_types)
    index = build_structural_index(data_config["kg_train"], no_evidence_pairs, entity_types=entity_types)

    rows = []
    reason_counts = Counter()
    no_gold_reason_counts = Counter()
    structural_counts = Counter()
    for pair in sorted(no_evidence_pairs, key=lambda item: (item.endpoint_id, item.drug_id, item.pair_id)):
        drug_out_edges = index["drug_out"].get(pair.drug_id, [])
        endpoint_in_edges = index["endpoint_in"].get(pair.endpoint_id, [])
        guided_drug_out_edges = index["guided_drug_out"].get(pair.drug_id, [])
        drug_out_nodes = {edge.tail for edge in drug_out_edges}
        endpoint_predecessors = {edge.head for edge in endpoint_in_edges}
        two_hop_bridges = sorted(drug_out_nodes & endpoint_predecessors)
        three_hop_bridge_count = _count_three_hop_bridges(
            drug_out_edges,
            endpoint_in_edges,
            index["mid_edges_by_head"],
        )
        has_gold_template = pair.pair_id in gold_pair_ids
        reason = classify_reason(
            has_gold_template=has_gold_template,
            drug_in_entity_table=pair.drug_id in entity_set,
            endpoint_in_entity_table=pair.endpoint_id in entity_set,
            drug_out_degree=index["out_degree"][pair.drug_id],
            endpoint_in_degree=index["in_degree"][pair.endpoint_id],
            guided_drug_out_degree=len(guided_drug_out_edges),
            two_hop_bridge_count=len(two_hop_bridges),
            three_hop_bridge_count=three_hop_bridge_count,
        )
        reason_counts[reason] += 1
        if not has_gold_template:
            no_gold_reason_counts[reason] += 1
        if drug_out_edges:
            structural_counts["pairs_with_drug_out_edges"] += 1
        if endpoint_in_edges:
            structural_counts["pairs_with_endpoint_in_edges"] += 1
        if two_hop_bridges:
            structural_counts["pairs_with_two_hop_bridge"] += 1
        if three_hop_bridge_count:
            structural_counts["pairs_with_three_hop_bridge"] += 1
        fallback_edge_count = len({edge.as_triple() for edge in drug_out_edges[:sample_limit] + endpoint_in_edges[:sample_limit]})
        if fallback_edge_count:
            structural_counts["pairs_with_fallback_neighbor_edges"] += 1

        rows.append(
            {
                "pair_id": pair.pair_id,
                "drug_id": pair.drug_id,
                "endpoint_id": pair.endpoint_id,
                "split": "train",
                "label": str(pair.label),
                "has_gold_template": str(has_gold_template).lower(),
                "evidence_path_count": str(evidence_counts.get(pair.pair_id, 0)),
                "no_evidence_scope": "gold_positive_already_supervised"
                if has_gold_template
                else "no_gold_positive_retrieval_miss",
                "drug_in_entity_table": str(pair.drug_id in entity_set).lower(),
                "endpoint_in_entity_table": str(pair.endpoint_id in entity_set).lower(),
                "drug_out_degree": str(index["out_degree"][pair.drug_id]),
                "drug_in_degree": str(index["in_degree"][pair.drug_id]),
                "endpoint_in_degree": str(index["in_degree"][pair.endpoint_id]),
                "endpoint_out_degree": str(index["out_degree"][pair.endpoint_id]),
                "guided_drug_out_degree": str(len(guided_drug_out_edges)),
                "two_hop_bridge_count": str(len(two_hop_bridges)),
                "three_hop_bridge_count": str(three_hop_bridge_count),
                "fallback_structural_edge_count": str(fallback_edge_count),
                "sample_drug_out_edges": _format_edges(drug_out_edges, sample_limit),
                "sample_endpoint_in_edges": _format_edges(endpoint_in_edges, sample_limit),
                "sample_two_hop_bridge_nodes": "|".join(two_hop_bridges[:sample_limit]),
                "reason_unretrieved": reason,
            }
        )

    output_path = Path(output_config["no_evidence_pairs"])
    summary_path = Path(output_config["summary_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NO_EVIDENCE_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    no_evidence_gold = sum(1 for pair in no_evidence_pairs if pair.pair_id in gold_pair_ids)
    no_evidence_no_gold = len(no_evidence_pairs) - no_evidence_gold
    summary = {
        "train_pairs": len(train_pairs),
        "train_positive_pairs": len(train_positive_pairs),
        "train_gold_template_positive_pairs": sum(1 for pair in train_positive_pairs if pair.pair_id in gold_pair_ids),
        "train_positive_pairs_with_evidence_paths": len(evidence_pair_ids),
        "train_positive_pairs_without_evidence_paths": len(no_evidence_pairs),
        "no_evidence_gold_template_pairs": no_evidence_gold,
        "no_evidence_no_gold_positive_pairs": no_evidence_no_gold,
        "reason_distribution_all_no_evidence": dict(sorted(reason_counts.items())),
        "reason_distribution_no_gold_no_evidence": dict(sorted(no_gold_reason_counts.items())),
        "structural_support_counts": dict(sorted(structural_counts.items())),
        "output_tsv": str(output_path),
        "leakage_checks": {
            "non_train_evidence_rows_seen": int(evidence_counts.get("_non_train_rows", 0)),
            "reported_valid_pairs": sum(1 for row in rows if row["split"] == "valid"),
            "reported_test_pairs": sum(1 for row in rows if row["split"] == "test"),
            "reported_negative_pairs": sum(1 for row in rows if row["label"] != "1"),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/no_evidence_report.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_no_evidence_report(load_config(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
