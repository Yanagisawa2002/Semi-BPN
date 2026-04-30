"""Build a top-K evidence-path subgraph for BioPathNet.

The output is a standard BioPathNet data directory. ``train1.txt`` contains the
union of edges from the top-K evidence paths for train pairs, while
``train2.txt``, ``valid.txt``, ``test.txt`` and entity metadata are copied from
the full BioPathNet directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from mechrep.data.no_evidence_report import Edge, read_train_kg_edges, sort_in_edges, sort_out_edges
from mechrep.templates.export_evidence_paths import load_entity_types


EVIDENCE_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "path_id",
    "path_node_ids",
    "path_node_types",
    "path_relation_types",
    "path_score",
)

BIOPATHNET_COPY_FILES = (
    "train2.txt",
    "valid.txt",
    "test.txt",
    "entity_names.txt",
    "entity_types.txt",
)


@dataclass(frozen=True)
class EvidencePath:
    pair_id: str
    drug_id: str
    endpoint_id: str
    split: str
    path_id: str
    node_ids: tuple[str, ...]
    node_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    path_score: float
    row_number: int

    def sort_key(self) -> tuple:
        return (-self.path_score, self.path_id, self.row_number)

    def edges(self) -> Iterable[tuple[str, str, str]]:
        for head, relation, tail in zip(self.node_ids, self.relation_types, self.node_ids[1:]):
            yield head, relation, tail


@dataclass(frozen=True)
class FallbackPair:
    pair_id: str
    drug_id: str
    endpoint_id: str
    has_gold_template: bool


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


def split_sequence(value: str, *, source: str, column: str) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        raise ValueError(f"{source} has empty {column}")
    parts = tuple(part.strip() for part in str(value).split("|"))
    if any(not part for part in parts):
        raise ValueError(f"{source} has empty token in {column}")
    return parts


def parse_evidence_row(row: dict[str, str], *, row_number: int, path: Path) -> EvidencePath:
    missing = [column for column in EVIDENCE_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    source = f"{path}:{row_number}"
    node_ids = split_sequence(row["path_node_ids"], source=source, column="path_node_ids")
    node_types = split_sequence(row["path_node_types"], source=source, column="path_node_types")
    relation_types = split_sequence(row["path_relation_types"], source=source, column="path_relation_types")
    if len(node_ids) < 2:
        raise ValueError(f"{source} path_node_ids must contain at least two nodes")
    if len(node_types) != len(node_ids):
        raise ValueError(f"{source} path_node_types length must equal path_node_ids length")
    if len(relation_types) != len(node_ids) - 1:
        raise ValueError(f"{source} path_relation_types length must equal path_node_ids length - 1")
    try:
        path_score = float(row["path_score"])
    except ValueError as exc:
        raise ValueError(f"{source} path_score must be numeric") from exc
    return EvidencePath(
        pair_id=row["pair_id"],
        drug_id=row["drug_id"],
        endpoint_id=row["endpoint_id"],
        split=row["split"],
        path_id=row["path_id"],
        node_ids=node_ids,
        node_types=node_types,
        relation_types=relation_types,
        path_score=path_score,
        row_number=row_number,
    )


def read_evidence_paths(path: str | Path, *, allowed_splits: set[str]) -> list[EvidencePath]:
    path = Path(path)
    rows: list[EvidencePath] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        missing = [column for column in EVIDENCE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            parsed = parse_evidence_row(row, row_number=row_number, path=path)
            if parsed.split in allowed_splits:
                rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} contains no evidence paths for splits {sorted(allowed_splits)}")
    return rows


def select_top_k_by_pair(paths: Sequence[EvidencePath], top_k: int) -> dict[str, list[EvidencePath]]:
    if top_k <= 0:
        raise ValueError("top_k_per_pair must be positive")
    grouped: dict[str, list[EvidencePath]] = defaultdict(list)
    for path in paths:
        grouped[path.pair_id].append(path)
    return {
        pair_id: sorted(pair_paths, key=lambda item: item.sort_key())[:top_k]
        for pair_id, pair_paths in sorted(grouped.items())
    }


def read_triples(path: str | Path) -> list[tuple[str, str, str]]:
    triples = []
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(f"{path}:{row_number} must have exactly three columns")
            triples.append((row[0], row[1], row[2]))
    return triples


def write_triples(path: str | Path, triples: Sequence[tuple[str, str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(triples)


def _parse_bool(value: str, *, source: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{source} must be a boolean value, got {value!r}")


def read_no_evidence_pairs(path: str | Path, *, pair_scope: str) -> list[FallbackPair]:
    path = Path(path)
    required = ["pair_id", "drug_id", "endpoint_id", "split", "label", "has_gold_template"]
    pairs: list[FallbackPair] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            source = f"{path}:{row_number}"
            if row["split"] != "train":
                raise ValueError(f"{source} fallback structural support only accepts train pairs")
            if row["label"] != "1":
                raise ValueError(f"{source} fallback structural support only accepts positive pairs")
            has_gold_template = _parse_bool(row["has_gold_template"], source=f"{source} has_gold_template")
            if pair_scope == "no_gold_only" and has_gold_template:
                continue
            if pair_scope == "gold_only" and not has_gold_template:
                continue
            if pair_scope not in {"all_no_evidence_train_positive", "no_gold_only", "gold_only"}:
                raise ValueError(
                    "fallback_structural_support.pair_scope must be one of "
                    "all_no_evidence_train_positive, no_gold_only, gold_only"
                )
            pairs.append(
                FallbackPair(
                    pair_id=row["pair_id"],
                    drug_id=row["drug_id"],
                    endpoint_id=row["endpoint_id"],
                    has_gold_template=has_gold_template,
                )
            )
    return sorted(pairs, key=lambda item: (item.endpoint_id, item.drug_id, item.pair_id))


def _add_non_target_edge(edge_set: set[tuple[str, str, str]], edge: Edge) -> bool:
    if edge.relation == "affects_endpoint":
        return False
    edge_set.add(edge.as_triple())
    return True


def _build_fallback_edge_index(
    kg_path: str | Path,
    pairs: Sequence[FallbackPair],
    *,
    entity_types: dict[str, str],
    include_three_hop_bridges: bool,
) -> dict:
    drugs = {pair.drug_id for pair in pairs}
    endpoints = {pair.endpoint_id for pair in pairs}
    drug_out = defaultdict(list)
    endpoint_in = defaultdict(list)

    for edge in read_train_kg_edges(kg_path):
        if edge.head in drugs:
            drug_out[edge.head].append(edge)
        if edge.tail in endpoints:
            endpoint_in[edge.tail].append(edge)

    sorted_drug_out = {node: sort_out_edges(edges, entity_types) for node, edges in drug_out.items()}
    sorted_endpoint_in = {node: sort_in_edges(edges, entity_types) for node, edges in endpoint_in.items()}

    mid_edges_by_head = {}
    if include_three_hop_bridges:
        first_hop_nodes = {edge.tail for edges in sorted_drug_out.values() for edge in edges}
        endpoint_predecessors = {edge.head for edges in sorted_endpoint_in.values() for edge in edges}
        raw_mid_edges = defaultdict(list)
        if first_hop_nodes and endpoint_predecessors:
            for edge in read_train_kg_edges(kg_path):
                if edge.head in first_hop_nodes and edge.tail in endpoint_predecessors:
                    raw_mid_edges[edge.head].append(edge)
        mid_edges_by_head = {
            node: sort_out_edges(edges, entity_types) for node, edges in raw_mid_edges.items()
        }

    return {
        "drug_out": sorted_drug_out,
        "endpoint_in": sorted_endpoint_in,
        "mid_edges_by_head": mid_edges_by_head,
    }


def build_fallback_structural_edges(
    source_train1: str | Path,
    config: dict,
) -> tuple[set[tuple[str, str, str]], dict]:
    if not config.get("enabled", False):
        return set(), {"enabled": False, "pseudo_labels_created": 0}

    no_evidence_pairs_path = config.get("no_evidence_pairs")
    if not no_evidence_pairs_path:
        raise ValueError("fallback_structural_support.enabled=true requires no_evidence_pairs")
    entity_types_path = config.get("entity_types")
    conversion_report_path = config.get("conversion_report")
    if not entity_types_path or not conversion_report_path:
        raise ValueError("fallback_structural_support requires entity_types and conversion_report")

    pair_scope = config.get("pair_scope", "all_no_evidence_train_positive")
    pairs = read_no_evidence_pairs(no_evidence_pairs_path, pair_scope=pair_scope)
    entity_types = load_entity_types(entity_types_path, conversion_report_path)
    include_two_hop_bridges = bool(config.get("include_two_hop_bridges", True))
    include_three_hop_bridges = bool(config.get("include_three_hop_bridges", True))
    include_neighbor_union = bool(config.get("include_neighbor_union", True))
    max_drug_edges_per_pair = int(config.get("max_drug_edges_per_pair", 50))
    max_endpoint_edges_per_pair = int(config.get("max_endpoint_edges_per_pair", 50))
    max_bridge_edges_per_pair = int(config.get("max_bridge_edges_per_pair", 100))

    index = _build_fallback_edge_index(
        source_train1,
        pairs,
        entity_types=entity_types,
        include_three_hop_bridges=include_three_hop_bridges,
    )
    fallback_edges: set[tuple[str, str, str]] = set()
    stats = Counter()
    per_pair_edge_counts = {}
    target_relation_edges_skipped = 0

    for pair in pairs:
        pair_edges: set[tuple[str, str, str]] = set()
        drug_edges = index["drug_out"].get(pair.drug_id, [])
        endpoint_edges = index["endpoint_in"].get(pair.endpoint_id, [])
        endpoint_edges_by_head = defaultdict(list)
        drug_edges_by_tail = defaultdict(list)
        for edge in endpoint_edges:
            endpoint_edges_by_head[edge.head].append(edge)
        for edge in drug_edges:
            drug_edges_by_tail[edge.tail].append(edge)

        if include_two_hop_bridges:
            bridges = sorted(set(drug_edges_by_tail) & set(endpoint_edges_by_head))
            bridge_edge_additions = 0
            found_two_hop = False
            for bridge_node in bridges:
                if bridge_edge_additions >= max_bridge_edges_per_pair:
                    break
                found_two_hop = True
                for edge in drug_edges_by_tail[bridge_node]:
                    if bridge_edge_additions >= max_bridge_edges_per_pair:
                        break
                    if edge.relation == "affects_endpoint":
                        target_relation_edges_skipped += 1
                        continue
                    pair_edges.add(edge.as_triple())
                    bridge_edge_additions += 1
            if found_two_hop:
                stats["fallback_pairs_with_two_hop_bridge"] += 1
                for edge in endpoint_edges_by_head[bridge_node]:
                    if bridge_edge_additions >= max_bridge_edges_per_pair:
                        break
                    if edge.relation == "affects_endpoint":
                        target_relation_edges_skipped += 1
                        continue
                    pair_edges.add(edge.as_triple())
                    bridge_edge_additions += 1

        if include_three_hop_bridges:
            endpoint_heads = set(endpoint_edges_by_head)
            bridge_edge_additions = 0
            found_three_hop = False
            for first_edge in drug_edges:
                if bridge_edge_additions >= max_bridge_edges_per_pair:
                    break
                for mid_edge in index["mid_edges_by_head"].get(first_edge.tail, []):
                    if bridge_edge_additions >= max_bridge_edges_per_pair:
                        break
                    if mid_edge.tail not in endpoint_heads:
                        continue
                    found_three_hop = True
                    for edge in (first_edge, mid_edge):
                        if bridge_edge_additions >= max_bridge_edges_per_pair:
                            break
                        if edge.relation == "affects_endpoint":
                            target_relation_edges_skipped += 1
                            continue
                        pair_edges.add(edge.as_triple())
                        bridge_edge_additions += 1
                    for endpoint_edge in endpoint_edges_by_head[mid_edge.tail]:
                        if bridge_edge_additions >= max_bridge_edges_per_pair:
                            break
                        if endpoint_edge.relation == "affects_endpoint":
                            target_relation_edges_skipped += 1
                            continue
                        pair_edges.add(endpoint_edge.as_triple())
                        bridge_edge_additions += 1
            if found_three_hop:
                stats["fallback_pairs_with_three_hop_bridge"] += 1

        if include_neighbor_union:
            for edge in drug_edges[:max_drug_edges_per_pair]:
                if not _add_non_target_edge(pair_edges, edge):
                    target_relation_edges_skipped += 1
            for edge in endpoint_edges[:max_endpoint_edges_per_pair]:
                if not _add_non_target_edge(pair_edges, edge):
                    target_relation_edges_skipped += 1

        fallback_edges.update(pair_edges)
        per_pair_edge_counts[pair.pair_id] = len(pair_edges)
        if pair_edges:
            stats["fallback_pairs_with_any_structural_edge"] += 1
        if pair.has_gold_template:
            stats["fallback_gold_pairs_considered"] += 1
        else:
            stats["fallback_no_gold_pairs_considered"] += 1

    stats["fallback_pairs_considered"] = len(pairs)
    stats["fallback_unique_edges_added"] = len(fallback_edges)
    stats["target_relation_edges_skipped"] = target_relation_edges_skipped
    stats["pseudo_labels_created"] = 0
    return fallback_edges, {
        "enabled": True,
        "no_evidence_pairs": str(no_evidence_pairs_path),
        "pair_scope": pair_scope,
        "include_two_hop_bridges": include_two_hop_bridges,
        "include_three_hop_bridges": include_three_hop_bridges,
        "include_neighbor_union": include_neighbor_union,
        "max_drug_edges_per_pair": max_drug_edges_per_pair,
        "max_endpoint_edges_per_pair": max_endpoint_edges_per_pair,
        "max_bridge_edges_per_pair": max_bridge_edges_per_pair,
        "stats": dict(sorted(stats.items())),
        "average_fallback_edges_per_pair": sum(per_pair_edge_counts.values()) / len(per_pair_edge_counts)
        if per_pair_edge_counts
        else None,
        "pairs_without_fallback_edges": sum(1 for count in per_pair_edge_counts.values() if count == 0),
    }


def read_gold_paths(path: str | Path) -> list[EvidencePath]:
    path = Path(path)
    rows: list[EvidencePath] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return rows
        required = [
            "pair_id",
            "drug_id",
            "endpoint_id",
            "path_id",
            "path_node_ids",
            "path_node_types",
            "path_relation_types",
            "split",
        ]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            pseudo_row = {
                "pair_id": row["pair_id"],
                "drug_id": row["drug_id"],
                "endpoint_id": row["endpoint_id"],
                "split": row["split"],
                "path_id": row["path_id"],
                "path_node_ids": row["path_node_ids"],
                "path_node_types": row["path_node_types"],
                "path_relation_types": row["path_relation_types"],
                "path_score": "1.0",
            }
            rows.append(parse_evidence_row(pseudo_row, row_number=row_number, path=path))
    return rows


def path_is_covered(path: EvidencePath, edge_set: set[tuple[str, str, str]]) -> bool:
    return all(edge in edge_set for edge in path.edges())


def compute_gold_path_coverage(
    gold_paths_file: str | Path | None,
    edge_set: set[tuple[str, str, str]],
) -> dict:
    if not gold_paths_file:
        return {}
    gold_paths = read_gold_paths(gold_paths_file)
    by_split = defaultdict(list)
    for path in gold_paths:
        by_split[path.split].append(path)
    report = {}
    for split, paths in sorted(by_split.items()):
        covered = sum(1 for path in paths if path_is_covered(path, edge_set))
        total_edges = sum(len(path.relation_types) for path in paths)
        covered_edges = sum(1 for path in paths for edge in path.edges() if edge in edge_set)
        report[split] = {
            "gold_paths": len(paths),
            "covered_gold_paths": covered,
            "gold_path_coverage": covered / len(paths) if paths else None,
            "gold_edges": total_edges,
            "covered_gold_edges": covered_edges,
            "gold_edge_coverage": covered_edges / total_edges if total_edges else None,
        }
    return report


def build_path_subgraph(config: dict) -> dict:
    source_dir = Path(config["source_dir"])
    evidence_paths = Path(config["evidence_paths_train"])
    output_dir = Path(config["output_dir"])
    top_k = int(config.get("top_k_per_pair", 50))
    allowed_splits = set(config.get("allowed_evidence_splits", ["train"]))
    include_train_gold_paths = bool(config.get("include_train_gold_paths", True))
    if allowed_splits != {"train"}:
        raise ValueError("Only train evidence paths are allowed for training subgraph construction by default")

    for file_name in ("train1.txt", *BIOPATHNET_COPY_FILES):
        if not (source_dir / file_name).exists():
            raise FileNotFoundError(f"Missing BioPathNet source file: {source_dir / file_name}")

    all_paths = read_evidence_paths(evidence_paths, allowed_splits=allowed_splits)
    selected_by_pair = select_top_k_by_pair(all_paths, top_k=top_k)
    selected_paths = [path for pair_paths in selected_by_pair.values() for path in pair_paths]
    edge_set = {edge for path in selected_paths for edge in path.edges()}

    gold_paths_file = config.get("gold_paths_file")
    train_gold_paths: list[EvidencePath] = []
    non_train_gold_paths_added = 0
    if include_train_gold_paths and gold_paths_file:
        all_gold_paths = read_gold_paths(gold_paths_file)
        train_gold_paths = [path for path in all_gold_paths if path.split == "train"]
        non_train_gold_paths_added = sum(1 for path in all_gold_paths if path.split != "train" and path in train_gold_paths)
        for path in train_gold_paths:
            edge_set.update(path.edges())

    fallback_edges, fallback_report = build_fallback_structural_edges(
        source_dir / "train1.txt",
        config.get("fallback_structural_support", {}),
    )
    edge_set.update(fallback_edges)

    sorted_edges = sorted(edge_set, key=lambda edge: (edge[0], edge[1], edge[2]))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_triples(output_dir / "train1.txt", sorted_edges)
    for file_name in BIOPATHNET_COPY_FILES:
        shutil.copy2(source_dir / file_name, output_dir / file_name)

    full_edges = read_triples(source_dir / "train1.txt")
    train_target_edges = read_triples(source_dir / "train2.txt")
    full_edge_set = set(full_edges)
    selected_nodes = {node for head, _, tail in edge_set for node in (head, tail)}
    full_nodes = {node for head, _, tail in full_edge_set for node in (head, tail)}
    relation_counts = Counter(relation for _, relation, _ in edge_set)
    path_lengths = [len(path.relation_types) for path in selected_paths]

    report = {
        "source_dir": str(source_dir),
        "evidence_paths_train": str(evidence_paths),
        "output_dir": str(output_dir),
        "top_k_per_pair": top_k,
        "allowed_evidence_splits": sorted(allowed_splits),
        "input_evidence_paths": len(all_paths),
        "pairs_with_evidence_paths": len(selected_by_pair),
        "train_target_edges": len(train_target_edges),
        "train_target_pair_coverage_by_evidence": len(selected_by_pair) / len(train_target_edges)
        if train_target_edges
        else None,
        "selected_evidence_paths": len(selected_paths),
        "include_train_gold_paths": include_train_gold_paths,
        "train_gold_paths_added": len(train_gold_paths),
        "fallback_structural_support": fallback_report,
        "selected_edges": len(edge_set),
        "selected_nodes": len(selected_nodes),
        "full_train1_edges": len(full_edge_set),
        "full_train1_nodes": len(full_nodes),
        "edge_reduction_factor": len(full_edge_set) / len(edge_set) if edge_set else None,
        "node_reduction_factor": len(full_nodes) / len(selected_nodes) if selected_nodes else None,
        "average_selected_paths_per_pair": len(selected_paths) / len(selected_by_pair),
        "average_selected_path_length": sum(path_lengths) / len(path_lengths) if path_lengths else None,
        "relation_distribution": dict(sorted(relation_counts.items())),
        "leakage_checks": {
            "non_train_evidence_paths_selected": sum(1 for path in selected_paths if path.split != "train"),
            "non_train_gold_paths_added": non_train_gold_paths_added,
            "target_relation_edges_in_train1": sum(1 for _, relation, _ in edge_set if relation == "affects_endpoint"),
            "pseudo_labels_created_by_fallback": fallback_report.get("stats", {}).get(
                "pseudo_labels_created", fallback_report.get("pseudo_labels_created", 0)
            ),
        },
        "gold_path_coverage": compute_gold_path_coverage(gold_paths_file, edge_set),
    }
    with (output_dir / "path_subgraph_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/path_subgraph_k50.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_path_subgraph(load_config(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
