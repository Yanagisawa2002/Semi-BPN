"""Export candidate evidence paths from the training KG for pseudo assignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

from mechrep.data.build_pairs import PairRecord, read_pair_tsv
from mechrep.data.gold_template_dataset import read_gold_template_label_tsv


EVIDENCE_PATH_COLUMNS = (
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

MECHANISM_NODE_TYPES = frozenset(
    {
        "BiologicalProcess",
        "MolecularFunction",
        "CellularComponent",
        "Phenotype",
        "Pathway",
    }
)

DEFAULT_GUIDED_SEED_RELATIONS = frozenset({"drug_protein", "drug_effect"})
DEFAULT_GUIDED_SEED_TYPES = frozenset({"Gene", "Protein", "GeneFamily", "Phenotype"})
DEFAULT_GUIDED_NODE_TYPES = frozenset(
    {
        "Gene",
        "Protein",
        "GeneFamily",
        "BiologicalProcess",
        "MolecularFunction",
        "CellularComponent",
        "Phenotype",
        "Pathway",
    }
)
DEFAULT_GUIDED_RELATIONS = frozenset(
    {
        "protein_protein",
        "bioprocess_protein",
        "molfunc_protein",
        "cellcomp_protein",
        "pathway_protein",
        "phenotype_protein",
        "bioprocess_bioprocess",
        "molfunc_molfunc",
        "cellcomp_cellcomp",
        "pathway_pathway",
        "phenotype_phenotype",
    }
)

RELATION_PRIORITY = {
    "drug_protein": 0,
    "drug_effect": 1,
    "protein_protein": 2,
    "bioprocess_protein": 3,
    "molfunc_protein": 4,
    "cellcomp_protein": 5,
    "pathway_protein": 6,
    "phenotype_protein": 7,
    "bioprocess_bioprocess": 8,
    "molfunc_molfunc": 9,
    "cellcomp_cellcomp": 10,
    "pathway_pathway": 11,
    "phenotype_phenotype": 12,
    "disease_protein": 13,
    "disease_phenotype_positive": 14,
    "disease_phenotype_negative": 15,
    "disease_disease": 16,
    "drug_drug": 50,
}

NODE_TYPE_PRIORITY = {
    "Gene": 0,
    "Protein": 1,
    "GeneFamily": 2,
    "BiologicalProcess": 3,
    "MolecularFunction": 4,
    "CellularComponent": 5,
    "Phenotype": 6,
    "Pathway": 7,
    "Disease": 8,
    "Drug": 30,
}


@dataclass(frozen=True)
class Edge:
    node: str
    relation: str


@dataclass(frozen=True)
class RetrievedPath:
    pair_id: str
    drug_id: str
    endpoint_id: str
    split: str
    node_ids: tuple[str, ...]
    node_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    path_score: float

    def mechanism_node_count(self) -> int:
        return sum(1 for node_type in self.node_types[1:-1] if node_type in MECHANISM_NODE_TYPES)

    def sort_key(self) -> tuple:
        return (
            -self.mechanism_node_count(),
            -self.path_score,
            len(self.relation_types),
            self.relation_types,
            self.node_types,
            self.node_ids,
        )

    def stable_path_id(self, rank: int) -> str:
        key = "||".join(self.node_ids) + "||" + "||".join(self.relation_types)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return f"EV_{self.pair_id}_{rank:03d}_{digest}"

    def as_row(self, rank: int) -> dict:
        return {
            "pair_id": self.pair_id,
            "drug_id": self.drug_id,
            "endpoint_id": self.endpoint_id,
            "split": self.split,
            "path_id": self.stable_path_id(rank),
            "path_node_ids": "|".join(self.node_ids),
            "path_node_types": "|".join(self.node_types),
            "path_relation_types": "|".join(self.relation_types),
            "path_score": f"{self.path_score:.8f}",
        }


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


@dataclass(frozen=True)
class RelationMapper:
    by_relation: Dict[str, str]
    by_context: Dict[tuple[str, str, str], str]

    @classmethod
    def identity(cls) -> "RelationMapper":
        return cls(by_relation={}, by_context={})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RelationMapper":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        by_relation = {
            str(source): str(target)
            for source, target in (payload.get("by_relation") or {}).items()
        }
        by_context = {}
        for index, item in enumerate(payload.get("by_context") or [], start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{path} by_context item {index} must be a mapping")
            required = ["source_type", "relation", "target_type", "mapped_relation"]
            missing = [key for key in required if key not in item or item[key] in (None, "")]
            if missing:
                raise ValueError(f"{path} by_context item {index} is missing keys: {missing}")
            by_context[(str(item["source_type"]), str(item["relation"]), str(item["target_type"]))] = str(
                item["mapped_relation"]
            )
        return cls(by_relation=by_relation, by_context=by_context)

    def map_relation(self, relation: str, source_type: str, target_type: str) -> str:
        return self.by_context.get((source_type, relation, target_type), self.by_relation.get(relation, relation))


def load_relation_mapper(config: dict) -> RelationMapper:
    mapping_config = config.get("relation_mapping", {})
    if not mapping_config or not mapping_config.get("enabled", False):
        return RelationMapper.identity()
    path = mapping_config.get("path")
    if not path:
        raise ValueError("relation_mapping.enabled=true requires relation_mapping.path")
    return RelationMapper.from_yaml(path)


def load_type_vocab(conversion_report_path: str | Path) -> dict[int, str]:
    with Path(conversion_report_path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    type_vocab = report.get("entity_report", {}).get("type_vocab")
    if not isinstance(type_vocab, dict) or not type_vocab:
        raise ValueError(f"{conversion_report_path} is missing entity_report.type_vocab")
    return {int(index): str(name) for name, index in type_vocab.items()}


def load_entity_types(entity_types_path: str | Path, conversion_report_path: str | Path) -> Dict[str, str]:
    index_to_type = load_type_vocab(conversion_report_path)
    entity_types = {}
    with Path(entity_types_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"{entity_types_path}:{row_number} must have exactly two columns")
            token, type_index = row
            index = int(type_index)
            if index not in index_to_type:
                raise ValueError(f"{entity_types_path}:{row_number} has unknown type index {index}")
            entity_types[token] = index_to_type[index]
    if not entity_types:
        raise ValueError(f"{entity_types_path} contains no entity types")
    return entity_types


def load_target_pairs(
    train_pairs_path: str | Path,
    gold_labels_train_path: str | Path,
    *,
    include_gold_pairs: bool = False,
    max_pairs: int | None = None,
) -> List[PairRecord]:
    records = read_pair_tsv(train_pairs_path, split="train")
    gold_pair_ids = set(read_gold_template_label_tsv(gold_labels_train_path))
    selected = [
        record
        for record in records
        if record.label == 1 and (include_gold_pairs or record.pair_id not in gold_pair_ids)
    ]
    selected.sort(key=lambda item: (item.endpoint_id, item.drug_id, item.pair_id))
    if max_pairs is not None:
        selected = selected[:max_pairs]
    if not selected:
        raise ValueError("No train positive pairs are available for evidence path retrieval")
    return selected


def read_train_kg_edges(path: str | Path) -> Iterable[tuple[str, str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(f"{path}:{row_number} must have exactly three tab-separated columns")
            yield row[0], row[1], row[2]


def _as_string_set(value: object, default: frozenset[str]) -> frozenset[str]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError("Retrieval set config values must be YAML lists")
    return frozenset(str(item) for item in value)


def edge_priority(edge: Edge, entity_types: Dict[str, str]) -> tuple:
    node_type = entity_types.get(edge.node, "")
    return (
        RELATION_PRIORITY.get(edge.relation, 100),
        NODE_TYPE_PRIORITY.get(node_type, 100),
        node_type,
        edge.node,
        edge.relation,
    )


def sort_and_cap_edges(
    edges_by_head: defaultdict[str, list[Edge]],
    *,
    entity_types: Dict[str, str],
    max_edges_per_node: int | None = None,
) -> dict[str, list[Edge]]:
    capped = {}
    for head, edges in edges_by_head.items():
        sorted_edges = sorted(edges, key=lambda edge: edge_priority(edge, entity_types))
        if max_edges_per_node is not None:
            sorted_edges = sorted_edges[:max_edges_per_node]
        capped[head] = sorted_edges
    return capped


def build_retrieval_index(
    kg_path: str | Path,
    pairs: Sequence[PairRecord],
    *,
    entity_types: Dict[str, str],
    retrieval_config: dict | None = None,
) -> dict:
    retrieval_config = retrieval_config or {}
    relevant_drugs = {pair.drug_id for pair in pairs}
    relevant_endpoints = {pair.endpoint_id for pair in pairs}
    guided_seed_relations = _as_string_set(
        retrieval_config.get("guided_seed_relations"),
        DEFAULT_GUIDED_SEED_RELATIONS,
    )
    guided_seed_types = _as_string_set(
        retrieval_config.get("guided_seed_node_types"),
        DEFAULT_GUIDED_SEED_TYPES,
    )
    guided_node_types = _as_string_set(
        retrieval_config.get("guided_node_types"),
        DEFAULT_GUIDED_NODE_TYPES,
    )
    guided_relations = _as_string_set(
        retrieval_config.get("guided_relations"),
        DEFAULT_GUIDED_RELATIONS,
    )
    max_guided_edges_per_node = retrieval_config.get("max_guided_edges_per_node", 80)
    if max_guided_edges_per_node is not None:
        max_guided_edges_per_node = int(max_guided_edges_per_node)

    drug_out = defaultdict(list)
    guided_drug_out = defaultdict(list)
    endpoint_in = defaultdict(list)
    guided_out = defaultdict(list)
    guided_reverse = defaultdict(list)

    for head, relation, tail in read_train_kg_edges(kg_path):
        head_type = entity_types.get(head)
        tail_type = entity_types.get(tail)
        if head in relevant_drugs:
            edge = Edge(tail, relation)
            drug_out[head].append(edge)
            if relation in guided_seed_relations and tail_type in guided_seed_types:
                guided_drug_out[head].append(edge)
        if tail in relevant_endpoints:
            endpoint_in[tail].append(Edge(head, relation))
        if relation in guided_relations and head_type in guided_node_types and tail_type in guided_node_types:
            edge = Edge(tail, relation)
            guided_out[head].append(edge)
            guided_reverse[tail].append(Edge(head, relation))

    first_hop_nodes = {edge.node for edges in drug_out.values() for edge in edges}
    guided_first_hop_nodes = {edge.node for edges in guided_drug_out.values() for edge in edges}
    endpoint_predecessors = {edge.node for edges in endpoint_in.values() for edge in edges}
    mid_out = defaultdict(list)
    if first_hop_nodes and endpoint_predecessors:
        for head, relation, tail in read_train_kg_edges(kg_path):
            if head in first_hop_nodes and tail in endpoint_predecessors:
                mid_out[head].append(Edge(tail, relation))

    return {
        "drug_out": sort_and_cap_edges(drug_out, entity_types=entity_types),
        "guided_drug_out": sort_and_cap_edges(guided_drug_out, entity_types=entity_types),
        "endpoint_in": sort_and_cap_edges(endpoint_in, entity_types=entity_types),
        "mid_out": sort_and_cap_edges(mid_out, entity_types=entity_types),
        "guided_out": sort_and_cap_edges(
            guided_out,
            entity_types=entity_types,
            max_edges_per_node=max_guided_edges_per_node,
        ),
        "guided_reverse": sort_and_cap_edges(
            guided_reverse,
            entity_types=entity_types,
            max_edges_per_node=max_guided_edges_per_node,
        ),
        "num_first_hop_nodes": len(first_hop_nodes),
        "num_guided_first_hop_nodes": len(guided_first_hop_nodes),
        "num_guided_reachable_nodes": len(guided_out),
        "num_endpoint_predecessors": len(endpoint_predecessors),
    }


def path_node_types(node_ids: Sequence[str], endpoint_id: str, entity_types: Dict[str, str]) -> tuple[str, ...]:
    types = []
    for index, node_id in enumerate(node_ids):
        if index == len(node_ids) - 1 and node_id == endpoint_id:
            types.append("Endpoint")
            continue
        if node_id not in entity_types:
            raise ValueError(f"Missing entity type for node {node_id!r}")
        types.append(entity_types[node_id])
    return tuple(types)


def make_path(
    pair: PairRecord,
    node_ids: Sequence[str],
    relation_types: Sequence[str],
    *,
    entity_types: Dict[str, str],
    relation_mapper: RelationMapper,
    mechanism_node_bonus: float = 0.15,
) -> RetrievedPath:
    if len(node_ids) != len(relation_types) + 1:
        raise ValueError(f"Path for pair {pair.pair_id!r} has inconsistent node/relation lengths")
    edge_count = len(relation_types)
    node_types = path_node_types(node_ids, pair.endpoint_id, entity_types)
    mechanism_node_count = sum(1 for node_type in node_types[1:-1] if node_type in MECHANISM_NODE_TYPES)
    # Shorter paths remain useful, but mechanism-rich paths should not be
    # crowded out by generic drug-drug bridges before template matching.
    path_score = (1.0 / edge_count) + (mechanism_node_bonus * mechanism_node_count)
    mapped_relations = tuple(
        relation_mapper.map_relation(relation, node_types[index], node_types[index + 1])
        for index, relation in enumerate(relation_types)
    )
    return RetrievedPath(
        pair_id=pair.pair_id,
        drug_id=pair.drug_id,
        endpoint_id=pair.endpoint_id,
        split=pair.split or "train",
        node_ids=tuple(node_ids),
        node_types=node_types,
        relation_types=mapped_relations,
        path_score=path_score,
    )


def retrieve_paths_for_pair(
    pair: PairRecord,
    *,
    drug_out: Dict[str, List[Edge]],
    guided_drug_out: Dict[str, List[Edge]],
    endpoint_in: Dict[str, List[Edge]],
    mid_out: Dict[str, List[Edge]],
    guided_out: Dict[str, List[Edge]],
    guided_reverse: Dict[str, List[Edge]],
    entity_types: Dict[str, str],
    relation_mapper: RelationMapper,
    max_paths_per_pair: int,
    max_first_hops_per_pair: int,
    max_guided_first_hops_per_pair: int,
    max_endpoint_predecessors_per_pair: int,
    max_guided_depth: int,
    max_candidate_paths_per_pair: int,
    mechanism_node_bonus: float,
    endpoint_reachability_cache: dict[str, dict[str, int]],
) -> List[RetrievedPath]:
    paths = []
    out_edges = drug_out.get(pair.drug_id, [])[:max_first_hops_per_pair]
    in_edges = endpoint_in.get(pair.endpoint_id, [])[:max_endpoint_predecessors_per_pair]
    pred_by_node = defaultdict(list)
    for edge in in_edges:
        pred_by_node[edge.node].append(edge.relation)

    for edge in out_edges:
        if edge.node == pair.endpoint_id:
            paths.append(
                make_path(
                    pair,
                    [pair.drug_id, pair.endpoint_id],
                    [edge.relation],
                    entity_types=entity_types,
                    relation_mapper=relation_mapper,
                    mechanism_node_bonus=mechanism_node_bonus,
                )
            )

    out_by_node = defaultdict(list)
    for edge in out_edges:
        out_by_node[edge.node].append(edge.relation)
    for mid_node in sorted(set(out_by_node) & set(pred_by_node)):
        for rel1 in sorted(out_by_node[mid_node]):
            for rel2 in sorted(pred_by_node[mid_node]):
                paths.append(
                    make_path(
                        pair,
                        [pair.drug_id, mid_node, pair.endpoint_id],
                        [rel1, rel2],
                        entity_types=entity_types,
                        relation_mapper=relation_mapper,
                        mechanism_node_bonus=mechanism_node_bonus,
                    )
                )

    pred_nodes = set(pred_by_node)
    for first_edge in out_edges:
        for middle_edge in mid_out.get(first_edge.node, []):
            if middle_edge.node not in pred_nodes:
                continue
            for rel3 in sorted(pred_by_node[middle_edge.node]):
                paths.append(
                    make_path(
                        pair,
                        [pair.drug_id, first_edge.node, middle_edge.node, pair.endpoint_id],
                        [first_edge.relation, middle_edge.relation, rel3],
                        entity_types=entity_types,
                        relation_mapper=relation_mapper,
                        mechanism_node_bonus=mechanism_node_bonus,
                    )
                )

    guided_seeds = guided_drug_out.get(pair.drug_id, [])[:max_guided_first_hops_per_pair]
    if pair.endpoint_id not in endpoint_reachability_cache:
        max_guided_hops_before_endpoint = max(0, max_guided_depth - 1)
        distances = {}
        frontier = set(pred_by_node)
        for node in frontier:
            distances[node] = 0
        for distance in range(1, max_guided_hops_before_endpoint + 1):
            next_frontier = set()
            for node in frontier:
                for reverse_edge in guided_reverse.get(node, []):
                    if reverse_edge.node in distances:
                        continue
                    distances[reverse_edge.node] = distance
                    next_frontier.add(reverse_edge.node)
            frontier = next_frontier
            if not frontier:
                break
        endpoint_reachability_cache[pair.endpoint_id] = distances
    reachable_distance = endpoint_reachability_cache[pair.endpoint_id]

    def add_endpoint_extensions(node_ids: list[str], relation_types: list[str]) -> None:
        if len(paths) >= max_candidate_paths_per_pair:
            return
        current_node = node_ids[-1]
        if current_node not in pred_by_node:
            return
        if len(relation_types) + 1 > max_guided_depth:
            return
        for endpoint_relation in sorted(pred_by_node[current_node]):
            paths.append(
                make_path(
                    pair,
                    [*node_ids, pair.endpoint_id],
                    [*relation_types, endpoint_relation],
                    entity_types=entity_types,
                    relation_mapper=relation_mapper,
                    mechanism_node_bonus=mechanism_node_bonus,
                )
            )

    def dfs(node_ids: list[str], relation_types: list[str]) -> None:
        if len(paths) >= max_candidate_paths_per_pair:
            return
        add_endpoint_extensions(node_ids, relation_types)
        # Reserve the final hop for current_node -> endpoint.
        if len(relation_types) >= max_guided_depth - 1:
            return
        current_node = node_ids[-1]
        remaining_guided_hops_after_next = max_guided_depth - len(relation_types) - 2
        for next_edge in guided_out.get(current_node, []):
            if len(paths) >= max_candidate_paths_per_pair:
                break
            if next_edge.node in node_ids:
                continue
            if reachable_distance.get(next_edge.node, max_guided_depth + 1) > remaining_guided_hops_after_next:
                continue
            dfs([*node_ids, next_edge.node], [*relation_types, next_edge.relation])

    for seed_edge in guided_seeds:
        if len(paths) >= max_candidate_paths_per_pair:
            break
        if seed_edge.node == pair.endpoint_id:
            continue
        if reachable_distance.get(seed_edge.node, max_guided_depth + 1) > max_guided_depth - 2:
            continue
        dfs([pair.drug_id, seed_edge.node], [seed_edge.relation])

    deduped = {}
    for path in paths:
        key = (path.node_ids, path.relation_types)
        deduped.setdefault(key, path)
    return sorted(deduped.values(), key=lambda item: item.sort_key())[:max_paths_per_pair]


def write_evidence_paths(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_PATH_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_evidence_paths(config: dict) -> dict:
    data_config = config["data"]
    retrieval_config = config.get("retrieval", {})
    output_path = Path(config["output"]["train_paths"])
    pairs = load_target_pairs(
        data_config["train_pairs"],
        data_config["gold_labels_train"],
        include_gold_pairs=bool(retrieval_config.get("include_gold_pairs", False)),
        max_pairs=retrieval_config.get("max_pairs"),
    )
    entity_types = load_entity_types(data_config["entity_types"], data_config["conversion_report"])
    relation_mapper = load_relation_mapper(config)
    index = build_retrieval_index(
        data_config["kg_train"],
        pairs,
        entity_types=entity_types,
        retrieval_config=retrieval_config,
    )

    max_paths_per_pair = int(retrieval_config.get("max_paths_per_pair", 20))
    max_first_hops_per_pair = int(retrieval_config.get("max_first_hops_per_pair", 200))
    max_guided_first_hops_per_pair = int(
        retrieval_config.get("max_guided_first_hops_per_pair", max_first_hops_per_pair)
    )
    max_endpoint_predecessors_per_pair = int(retrieval_config.get("max_endpoint_predecessors_per_pair", 200))
    max_guided_depth = int(retrieval_config.get("max_guided_depth", 5))
    max_candidate_paths_per_pair = int(retrieval_config.get("max_candidate_paths_per_pair", max_paths_per_pair * 50))
    mechanism_node_bonus = float(retrieval_config.get("mechanism_node_bonus", 0.15))

    rows = []
    pairs_with_paths = 0
    per_pair_counts = {}
    endpoint_reachability_cache = {}
    for pair in pairs:
        paths = retrieve_paths_for_pair(
            pair,
            drug_out=index["drug_out"],
            guided_drug_out=index["guided_drug_out"],
            endpoint_in=index["endpoint_in"],
            mid_out=index["mid_out"],
            guided_out=index["guided_out"],
            guided_reverse=index["guided_reverse"],
            entity_types=entity_types,
            relation_mapper=relation_mapper,
            max_paths_per_pair=max_paths_per_pair,
            max_first_hops_per_pair=max_first_hops_per_pair,
            max_guided_first_hops_per_pair=max_guided_first_hops_per_pair,
            max_endpoint_predecessors_per_pair=max_endpoint_predecessors_per_pair,
            max_guided_depth=max_guided_depth,
            max_candidate_paths_per_pair=max_candidate_paths_per_pair,
            mechanism_node_bonus=mechanism_node_bonus,
            endpoint_reachability_cache=endpoint_reachability_cache,
        )
        if paths:
            pairs_with_paths += 1
        per_pair_counts[pair.pair_id] = len(paths)
        for rank, path in enumerate(paths, start=1):
            rows.append(path.as_row(rank))

    write_evidence_paths(output_path, rows)
    report = {
        "target_train_positive_no_gold_pairs": len(pairs),
        "pairs_with_candidate_paths": pairs_with_paths,
        "pairs_without_candidate_paths": len(pairs) - pairs_with_paths,
        "candidate_paths_written": len(rows),
        "max_paths_per_pair": max_paths_per_pair,
        "max_first_hops_per_pair": max_first_hops_per_pair,
        "max_guided_first_hops_per_pair": max_guided_first_hops_per_pair,
        "max_endpoint_predecessors_per_pair": max_endpoint_predecessors_per_pair,
        "max_guided_depth": max_guided_depth,
        "max_candidate_paths_per_pair": max_candidate_paths_per_pair,
        "mechanism_node_bonus": mechanism_node_bonus,
        "num_first_hop_nodes": index["num_first_hop_nodes"],
        "num_guided_first_hop_nodes": index["num_guided_first_hop_nodes"],
        "num_guided_reachable_nodes": index["num_guided_reachable_nodes"],
        "num_endpoint_reachability_cache_entries": len(endpoint_reachability_cache),
        "num_endpoint_predecessors": index["num_endpoint_predecessors"],
        "relation_mapping_enabled": bool(config.get("relation_mapping", {}).get("enabled", False)),
        "relation_mapping_path": config.get("relation_mapping", {}).get("path"),
        "output_path": str(output_path),
    }
    report_path = output_path.with_name(output_path.stem + "_report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evidence_paths.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_evidence_paths(load_config(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
