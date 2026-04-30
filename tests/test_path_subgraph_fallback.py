import csv
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.build_path_subgraph import build_path_subgraph


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"path_subgraph_fallback_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _read_triples(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {tuple(row) for row in csv.reader(handle, delimiter="\t")}


def test_path_subgraph_fallback_adds_structure_without_target_or_pseudo_labels():
    tmp_dir = _workspace_tmp_dir()
    try:
        source_dir = tmp_dir / "full"
        out_dir = tmp_dir / "fallback"
        _write_tsv(
            source_dir / "train1.txt",
            [
                ["D0", "drug_protein", "G0"],
                ["G0", "disease_protein", "E0"],
                ["D1", "drug_protein", "G1"],
                ["G1", "disease_protein", "E1"],
                ["D1", "affects_endpoint", "E1"],
            ],
        )
        _write_tsv(source_dir / "train2.txt", [["D0", "affects_endpoint", "E0"], ["D1", "affects_endpoint", "E1"]])
        _write_tsv(source_dir / "valid.txt", [["D2", "affects_endpoint", "E2"]])
        _write_tsv(source_dir / "test.txt", [["D3", "affects_endpoint", "E3"]])
        _write_tsv(source_dir / "entity_names.txt", [["D0", "D0"], ["D1", "D1"], ["G0", "G0"], ["G1", "G1"], ["E0", "E0"], ["E1", "E1"]])
        _write_tsv(source_dir / "entity_types.txt", [["D0", 0], ["D1", 0], ["G0", 1], ["G1", 1], ["E0", 2], ["E1", 2]])
        (tmp_dir / "conversion_report.json").write_text(
            '{"entity_report":{"type_vocab":{"Drug":0,"Gene":1,"Disease":2}}}',
            encoding="utf-8",
        )
        _write_tsv(
            tmp_dir / "evidence.tsv",
            [
                [
                    "pair_id",
                    "drug_id",
                    "endpoint_id",
                    "split",
                    "path_id",
                    "path_node_ids",
                    "path_node_types",
                    "path_relation_types",
                    "path_score",
                ],
                ["P0", "D0", "E0", "train", "EV0", "D0|G0|E0", "Drug|Gene|Endpoint", "drug_protein|disease_protein", "1.0"],
            ],
        )
        _write_tsv(
            tmp_dir / "no_evidence.tsv",
            [
                ["pair_id", "drug_id", "endpoint_id", "split", "label", "has_gold_template"],
                ["P1", "D1", "E1", "train", "1", "false"],
            ],
        )

        report = build_path_subgraph(
            {
                "source_dir": str(source_dir),
                "evidence_paths_train": str(tmp_dir / "evidence.tsv"),
                "output_dir": str(out_dir),
                "top_k_per_pair": 1,
                "allowed_evidence_splits": ["train"],
                "include_train_gold_paths": False,
                "fallback_structural_support": {
                    "enabled": True,
                    "no_evidence_pairs": str(tmp_dir / "no_evidence.tsv"),
                    "entity_types": str(source_dir / "entity_types.txt"),
                    "conversion_report": str(tmp_dir / "conversion_report.json"),
                    "include_neighbor_union": True,
                    "include_two_hop_bridges": True,
                    "include_three_hop_bridges": False,
                    "max_drug_edges_per_pair": 10,
                    "max_endpoint_edges_per_pair": 10,
                    "max_bridge_edges_per_pair": 10,
                },
            }
        )

        triples = _read_triples(out_dir / "train1.txt")
        assert ("D1", "drug_protein", "G1") in triples
        assert ("G1", "disease_protein", "E1") in triples
        assert ("D1", "affects_endpoint", "E1") not in triples
        fallback_stats = report["fallback_structural_support"]["stats"]
        assert fallback_stats["fallback_pairs_considered"] == 1
        assert fallback_stats["pseudo_labels_created"] == 0
        assert report["leakage_checks"]["target_relation_edges_in_train1"] == 0
        assert report["leakage_checks"]["pseudo_labels_created_by_fallback"] == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
