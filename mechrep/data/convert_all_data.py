"""Convert the all_data(indication_only) bundle into mechrep/BioPathNet inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

from mechrep.data.biopathnet_adapter import EndpointBioPathNetAdapter
from mechrep.data.build_ood_splits import build_ood_splits


DIRECT_PREDICTION_RELATIONS = {"indication"}
PAIR_COLUMNS = ("pair_id", "drug_id", "endpoint_id", "label")
GOLD_PATH_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "path_id",
    "path_node_ids",
    "path_node_types",
    "path_relation_types",
    "split",
)


def stable_pair_id(drug_id: str, endpoint_id: str) -> str:
    digest = hashlib.sha1(f"{drug_id}||{endpoint_id}".encode("utf-8")).hexdigest()[:16]
    return f"PAIR_{digest}"


def canonical_entity_id(entity_type: str, source: str, entity_id: str) -> str:
    if not entity_type or not source or not entity_id:
        raise ValueError(f"Cannot build entity id from type={entity_type!r}, source={source!r}, id={entity_id!r}")
    token = f"{entity_type}:{source}:{entity_id}"
    if re.search(r"\s", token):
        raise ValueError(f"Entity token contains whitespace and cannot be used by BioPathNet: {token!r}")
    return token


def normalize_node_type(primekg_type: str, *, is_endpoint: bool = False) -> str:
    if is_endpoint:
        return "Endpoint"
    mapping = {
        "drug": "Drug",
        "gene/protein": "Gene",
        "disease": "Disease",
        "pathway": "Pathway",
        "biological_process": "BiologicalProcess",
        "chemical_substance": "ChemicalSubstance",
        "effect/phenotype": "Phenotype",
        "exposure": "Exposure",
        "anatomy": "Anatomy",
    }
    return mapping.get(primekg_type, primekg_type.replace("/", "_").title().replace("_", ""))


def normalize_relation_type(relation: str) -> str:
    relation = relation.strip().lower()
    relation = re.sub(r"[^a-z0-9]+", "_", relation)
    relation = re.sub(r"_+", "_", relation).strip("_")
    if not relation:
        raise ValueError("Encountered empty relation type after normalization")
    return relation


def write_tsv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def load_mechanism_graphs(mech_json_path: str | Path) -> Dict[str, dict]:
    path = Path(mech_json_path)
    with path.open("r", encoding="utf-8") as handle:
        graphs = json.load(handle)
    if not isinstance(graphs, list):
        raise ValueError(f"{path} must contain a list of mechanism graph objects")
    by_path_id = {}
    for item in graphs:
        path_id = item.get("graph", {}).get("_id")
        if not path_id:
            raise ValueError(f"{path} contains a graph without graph._id")
        if path_id in by_path_id:
            raise ValueError(f"Duplicate mechanism graph id: {path_id}")
        by_path_id[path_id] = item
    return by_path_id


def graph_endpoint_tokens(graph: dict) -> tuple[str, str] | None:
    drugbank = graph.get("graph", {}).get("drugbank", "")
    drugbank_id = drugbank.split(":", 1)[-1] if drugbank else ""
    disease_mesh = graph.get("graph", {}).get("disease_mesh", "")
    drug_token = None
    endpoint_token = None
    for node in graph.get("nodes", []):
        mapping = node.get("primekg_mapping", {})
        if not mapping.get("mapped", False):
            continue
        token = canonical_entity_id(
            mapping.get("primekg_type", ""),
            mapping.get("primekg_source", ""),
            mapping.get("primekg_id", ""),
        )
        if mapping.get("primekg_type") == "drug" and mapping.get("primekg_id") == drugbank_id:
            drug_token = token
        if node.get("id") == disease_mesh and mapping.get("primekg_type") == "disease":
            endpoint_token = token
    if not drug_token or not endpoint_token:
        return None
    return drug_token, endpoint_token


def collect_mechanism_positive_pairs(mech_json_path: str | Path) -> tuple[set[tuple[str, str]], dict]:
    by_path_id = load_mechanism_graphs(mech_json_path)
    pairs = set()
    missing = []
    for path_id, graph in by_path_id.items():
        tokens = graph_endpoint_tokens(graph)
        if tokens is None:
            missing.append(path_id)
            continue
        pairs.add(tokens)
    return pairs, {"mechanism_graphs_without_endpoint_mapping": missing}


def scan_kg_and_write_train1(
    kg_path: str | Path,
    train1_path: str | Path,
    *,
    remove_prediction_edges: bool = True,
) -> dict:
    kg_path = Path(kg_path)
    train1_path = Path(train1_path)
    train1_path.parent.mkdir(parents=True, exist_ok=True)

    entity_types = {}
    entity_names = {}
    positive_pairs = set()
    clinical_prediction_edges = 0
    written_edges = 0
    relation_counts = Counter()
    type_conflicts = []

    with kg_path.open("r", newline="", encoding="utf-8") as source, train1_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        for row in reader:
            head = canonical_entity_id(row["x_type"], row["x_source"], row["x_id"])
            tail = canonical_entity_id(row["y_type"], row["y_source"], row["y_id"])
            relation = normalize_relation_type(row["relation"] or row["display_relation"])

            for token, node_type, name in [
                (head, row["x_type"], row["x_name"]),
                (tail, row["y_type"], row["y_name"]),
            ]:
                normalized_type = normalize_node_type(node_type)
                if token in entity_types and entity_types[token] != normalized_type:
                    type_conflicts.append((token, entity_types[token], normalized_type))
                entity_types[token] = normalized_type
                entity_names.setdefault(token, name)

            is_prediction_edge = (
                {row["x_type"], row["y_type"]} == {"drug", "disease"}
                and row["display_relation"] in DIRECT_PREDICTION_RELATIONS
            )
            if is_prediction_edge:
                if row["x_type"] == "drug":
                    positive_pairs.add((head, tail))
                else:
                    positive_pairs.add((tail, head))
                clinical_prediction_edges += 1
                if remove_prediction_edges:
                    continue

            writer.writerow([head, relation, tail])
            relation_counts[relation] += 1
            written_edges += 1

    if type_conflicts:
        raise ValueError(f"Entity type conflicts found, first conflicts: {type_conflicts[:5]}")

    return {
        "entity_types": entity_types,
        "entity_names": entity_names,
        "kg_positive_pairs": positive_pairs,
        "clinical_prediction_edges": clinical_prediction_edges,
        "train1_edges_written": written_edges,
        "relation_counts_top20": relation_counts.most_common(20),
    }


def sample_negative_pairs(
    positive_pairs: set[tuple[str, str]],
    *,
    negative_ratio: float,
    seed: int,
) -> set[tuple[str, str]]:
    if negative_ratio <= 0:
        return set()
    drugs = sorted({drug for drug, _ in positive_pairs})
    endpoints = sorted({endpoint for _, endpoint in positive_pairs})
    endpoint_to_positive_drugs = defaultdict(set)
    for drug, endpoint in positive_pairs:
        endpoint_to_positive_drugs[endpoint].add(drug)

    rng = random.Random(seed)
    negatives = set()
    for endpoint in endpoints:
        positives_for_endpoint = sorted(endpoint_to_positive_drugs[endpoint])
        candidates = [drug for drug in drugs if drug not in endpoint_to_positive_drugs[endpoint]]
        if not candidates:
            raise ValueError(f"Cannot sample negatives for endpoint {endpoint}: no candidate drugs remain")
        target_count = max(1, round(len(positives_for_endpoint) * negative_ratio))
        if target_count > len(candidates):
            raise ValueError(
                f"Cannot sample {target_count} negatives for endpoint {endpoint}: only {len(candidates)} candidates"
            )
        for drug in rng.sample(candidates, target_count):
            negatives.add((drug, endpoint))
    return negatives


def pair_rows_from_sets(
    positives: set[tuple[str, str]],
    negatives: set[tuple[str, str]],
) -> tuple[List[dict], Dict[tuple[str, str], str]]:
    pair_id_by_pair = {}
    rows = []
    for label, pairs in [(1, positives), (0, negatives)]:
        for drug, endpoint in sorted(pairs):
            pair_id = stable_pair_id(drug, endpoint)
            if pair_id in pair_id_by_pair.values():
                raise ValueError(f"Pair id collision for {drug}, {endpoint}: {pair_id}")
            pair_id_by_pair[(drug, endpoint)] = pair_id
            rows.append(
                {
                    "pair_id": pair_id,
                    "drug_id": drug,
                    "endpoint_id": endpoint,
                    "label": str(label),
                }
            )
    rows.sort(key=lambda row: (row["endpoint_id"], row["drug_id"], row["label"]))
    return rows, pair_id_by_pair


def write_entity_files(output_dir: str | Path, entity_types: dict, entity_names: dict) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    type_vocab = {node_type: index for index, node_type in enumerate(sorted(set(entity_types.values())))}
    with (output_dir / "entity_types.txt").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for token in sorted(entity_types):
            writer.writerow([token, type_vocab[entity_types[token]]])
    with (output_dir / "entity_names.txt").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for token in sorted(entity_types):
            writer.writerow([token, entity_names.get(token, token)])
    return {"num_entities": len(entity_types), "num_entity_types": len(type_vocab), "type_vocab": type_vocab}


def linearize_graph(graph: dict) -> tuple[list[str], list[dict]] | tuple[None, str]:
    nodes = {node["id"] for node in graph.get("nodes", [])}
    links = graph.get("links", [])
    if len(nodes) < 2:
        return None, "too_few_nodes"
    if len(links) != len(nodes) - 1:
        return None, "non_linear_edge_count"

    out_links = defaultdict(list)
    indeg = Counter()
    outdeg = Counter()
    for link in links:
        source = link.get("source")
        target = link.get("target")
        if source not in nodes or target not in nodes:
            return None, "link_references_missing_node"
        out_links[source].append(link)
        outdeg[source] += 1
        indeg[target] += 1

    starts = [node for node in nodes if indeg[node] == 0]
    ends = [node for node in nodes if outdeg[node] == 0]
    if len(starts) != 1 or len(ends) != 1:
        return None, "non_linear_degree"

    start = starts[0]
    end = ends[0]
    order = [start]
    ordered_links = []
    current = start
    seen = {current}
    while current != end:
        if len(out_links[current]) != 1:
            return None, "non_linear_branch"
        link = out_links[current][0]
        ordered_links.append(link)
        current = link["target"]
        if current in seen:
            return None, "cycle"
        seen.add(current)
        order.append(current)

    if len(order) != len(nodes):
        return None, "disconnected"
    return order, ordered_links


def convert_gold_paths(
    mech_csv_path: str | Path,
    mech_json_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path,
    *,
    pair_id_by_pair: Dict[tuple[str, str], str],
    split_by_pair_id: Dict[str, str],
) -> dict:
    graphs = load_mechanism_graphs(mech_json_path)
    output_path = Path(output_path)
    rejected_path = Path(rejected_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    rejected_reasons = Counter()
    converted = 0
    total = 0
    with Path(mech_csv_path).open("r", newline="", encoding="utf-8") as csv_handle, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as out_handle, rejected_path.open("w", newline="", encoding="utf-8") as reject_handle:
        reader = csv.DictReader(csv_handle)
        writer = csv.DictWriter(out_handle, fieldnames=GOLD_PATH_COLUMNS, delimiter="\t", lineterminator="\n")
        reject_writer = csv.DictWriter(
            reject_handle,
            fieldnames=["path_id", "reason", "drug_id", "endpoint_id"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        reject_writer.writeheader()
        for row in reader:
            total += 1
            path_id = row["path_id"]
            graph = graphs.get(path_id)
            if graph is None:
                rejected_reasons["missing_json_graph"] += 1
                reject_writer.writerow({"path_id": path_id, "reason": "missing_json_graph", "drug_id": "", "endpoint_id": ""})
                continue

            tokens = graph_endpoint_tokens(graph)
            if tokens is None:
                rejected_reasons["missing_endpoint_mapping"] += 1
                reject_writer.writerow({"path_id": path_id, "reason": "missing_endpoint_mapping", "drug_id": "", "endpoint_id": ""})
                continue
            drug_id, endpoint_id = tokens
            pair_id = pair_id_by_pair.get((drug_id, endpoint_id))
            if pair_id is None:
                rejected_reasons["pair_not_in_pair_table"] += 1
                reject_writer.writerow(
                    {"path_id": path_id, "reason": "pair_not_in_pair_table", "drug_id": drug_id, "endpoint_id": endpoint_id}
                )
                continue

            linear_result, ordered_links_or_reason = linearize_graph(graph)
            if linear_result is None:
                reason = str(ordered_links_or_reason)
                rejected_reasons[reason] += 1
                reject_writer.writerow({"path_id": path_id, "reason": reason, "drug_id": drug_id, "endpoint_id": endpoint_id})
                continue
            node_order = linear_result
            ordered_links = ordered_links_or_reason
            node_by_original_id = {node["id"]: node for node in graph["nodes"]}

            try:
                node_ids = []
                node_types = []
                for original_id in node_order:
                    node = node_by_original_id[original_id]
                    mapping = node.get("primekg_mapping", {})
                    token = canonical_entity_id(
                        mapping.get("primekg_type", ""),
                        mapping.get("primekg_source", ""),
                        mapping.get("primekg_id", ""),
                    )
                    node_ids.append(token)
                    node_types.append(normalize_node_type(mapping.get("primekg_type", ""), is_endpoint=(token == endpoint_id)))
                relation_types = [normalize_relation_type(link.get("key", "")) for link in ordered_links]
            except ValueError as exc:
                rejected_reasons["canonicalization_error"] += 1
                reject_writer.writerow(
                    {"path_id": path_id, "reason": f"canonicalization_error:{exc}", "drug_id": drug_id, "endpoint_id": endpoint_id}
                )
                continue

            if node_ids[0] != drug_id or node_ids[-1] != endpoint_id:
                rejected_reasons["linear_path_not_drug_to_endpoint"] += 1
                reject_writer.writerow(
                    {
                        "path_id": path_id,
                        "reason": "linear_path_not_drug_to_endpoint",
                        "drug_id": drug_id,
                        "endpoint_id": endpoint_id,
                    }
                )
                continue

            writer.writerow(
                {
                    "pair_id": pair_id,
                    "drug_id": drug_id,
                    "endpoint_id": endpoint_id,
                    "path_id": path_id,
                    "path_node_ids": "|".join(node_ids),
                    "path_node_types": "|".join(node_types),
                    "path_relation_types": "|".join(relation_types),
                    "split": split_by_pair_id[pair_id],
                }
            )
            converted += 1

    return {
        "gold_paths_total": total,
        "gold_paths_converted_linear": converted,
        "gold_paths_rejected": total - converted,
        "gold_path_rejection_reasons": dict(sorted(rejected_reasons.items())),
    }


def load_split_by_pair_id(split_dir: str | Path) -> Dict[str, str]:
    split_by_pair_id = {}
    for split in ["train", "valid", "test"]:
        with (Path(split_dir) / f"{split}.tsv").open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                pair_id = row["pair_id"]
                if pair_id in split_by_pair_id:
                    raise ValueError(f"pair_id appears in multiple split files: {pair_id}")
                split_by_pair_id[pair_id] = split
    return split_by_pair_id


def run_conversion(config: dict) -> dict:
    all_data_dir = Path(config.get("all_data_dir", "all_data(indication_only)"))
    output_dir = Path(config.get("output_dir", "data/all_data_unified"))
    seed = int(config.get("seed", 1024))
    negative_ratio = float(config.get("negative_ratio", 1.0))
    valid_ratio = float(config.get("valid_ratio", 0.1))
    test_ratio = float(config.get("test_ratio", 0.2))
    remove_prediction_edges = bool(config.get("remove_prediction_edges_from_kg", True))

    kg_path = all_data_dir / "forSemi_KG.csv"
    mech_csv_path = all_data_dir / "forSemi_Mech.csv"
    mech_json_path = all_data_dir / "forSemi_Mech.json"
    for path in [kg_path, mech_csv_path, mech_json_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required all-data file: {path}")

    biopathnet_dir = output_dir / "biopathnet"
    train1_path = biopathnet_dir / "train1.txt"
    scan_report = scan_kg_and_write_train1(
        kg_path,
        train1_path,
        remove_prediction_edges=remove_prediction_edges,
    )
    mech_pairs, mech_pair_report = collect_mechanism_positive_pairs(mech_json_path)
    positive_pairs = set(scan_report["kg_positive_pairs"]) | mech_pairs
    negative_pairs = sample_negative_pairs(positive_pairs, negative_ratio=negative_ratio, seed=seed)
    pair_rows, pair_id_by_pair = pair_rows_from_sets(positive_pairs, negative_pairs)

    pairs_path = output_dir / "pairs.tsv"
    write_tsv(pairs_path, PAIR_COLUMNS, pair_rows)
    splits_dir = output_dir / "splits"
    leakage_report = build_ood_splits(
        pairs_path,
        splits_dir,
        split_type="endpoint_ood",
        seed=seed,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    adapter = EndpointBioPathNetAdapter.from_split_dir(splits_dir)
    adapter_counts = adapter.write_biopathnet_triples(biopathnet_dir, positive_only=True)
    entity_report = write_entity_files(biopathnet_dir, scan_report["entity_types"], scan_report["entity_names"])
    split_by_pair_id = load_split_by_pair_id(splits_dir)
    gold_report = convert_gold_paths(
        mech_csv_path,
        mech_json_path,
        output_dir / "gold_paths.tsv",
        output_dir / "gold_paths_rejected.tsv",
        pair_id_by_pair=pair_id_by_pair,
        split_by_pair_id=split_by_pair_id,
    )

    report = {
        "all_data_dir": str(all_data_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "negative_ratio": negative_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "remove_prediction_edges_from_kg": remove_prediction_edges,
        "kg_positive_pairs": len(scan_report["kg_positive_pairs"]),
        "mechanism_positive_pairs": len(mech_pairs),
        "positive_pairs_union": len(positive_pairs),
        "negative_pairs_sampled": len(negative_pairs),
        "pair_rows_total": len(pair_rows),
        "clinical_prediction_edges_in_kg": scan_report["clinical_prediction_edges"],
        "train1_edges_written": scan_report["train1_edges_written"],
        "adapter_positive_triple_counts": adapter_counts,
        "entity_report": entity_report,
        "leakage_report": leakage_report,
        "mechanism_pair_report": mech_pair_report,
        "gold_path_report": gold_report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} must contain a mapping")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/all_data_conversion.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--negative-ratio", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.negative_ratio is not None:
        config["negative_ratio"] = args.negative_ratio
    report = run_conversion(config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
