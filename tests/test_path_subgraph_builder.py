import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.build_path_subgraph import build_path_subgraph


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"path_subgraph_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _read_triples(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [tuple(row) for row in csv.reader(handle, delimiter="\t")]


def _read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [tuple(row) for row in csv.reader(handle, delimiter="\t")]


def test_build_path_subgraph_top_k_deduplicates_and_reports_coverage():
    tmp_dir = _workspace_tmp_dir()
    try:
        source_dir = tmp_dir / "full"
        out_dir = tmp_dir / "subgraph"
        _write_tsv(
            source_dir / "train1.txt",
            [
                ["D1", "r1", "G1"],
                ["G1", "r2", "E1"],
                ["D1", "r3", "G2"],
                ["G2", "r4", "E1"],
                ["D2", "r5", "G3"],
            ],
        )
        for name in ("train2.txt", "valid.txt", "test.txt"):
            _write_tsv(source_dir / name, [["D1", "affects_endpoint", "E1"]])
        _write_tsv(
            source_dir / "entity_names.txt",
            [["D1", "D1"], ["G1", "G1"], ["E1", "E1"], ["UNUSED", "unused"]],
        )
        _write_tsv(source_dir / "entity_types.txt", [["D1", 0], ["G1", 1], ["E1", 2], ["UNUSED", 9]])

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
                ["P1", "D1", "E1", "train", "p_low", "D1|G2|E1", "Drug|Gene|Endpoint", "r3|r4", "0.1"],
                ["P1", "D1", "E1", "train", "p_high", "D1|G1|E1", "Drug|Gene|Endpoint", "r1|r2", "0.9"],
                ["P2", "D2", "E2", "valid", "p_bad", "D2|G3|E2", "Drug|Gene|Endpoint", "r5|r6", "1.0"],
            ],
        )
        _write_tsv(
            tmp_dir / "gold.tsv",
            [
                [
                    "pair_id",
                    "drug_id",
                    "endpoint_id",
                    "path_id",
                    "path_node_ids",
                    "path_node_types",
                    "path_relation_types",
                    "split",
                ],
                ["P1", "D1", "E1", "g1", "D1|G1|E1", "Drug|Gene|Endpoint", "r1|r2", "train"],
                ["P2", "D2", "E2", "g_valid", "D2|G3|E2", "Drug|Gene|Endpoint", "r5|r6", "valid"],
            ],
        )

        report = build_path_subgraph(
            {
                "source_dir": str(source_dir),
                "evidence_paths_train": str(tmp_dir / "evidence.tsv"),
                "gold_paths_file": str(tmp_dir / "gold.tsv"),
                "output_dir": str(out_dir),
                "top_k_per_pair": 1,
                "allowed_evidence_splits": ["train"],
            }
        )

        assert _read_triples(out_dir / "train1.txt") == [("D1", "r1", "G1"), ("G1", "r2", "E1")]
        assert (out_dir / "train2.txt").exists()
        assert report["selected_evidence_paths"] == 1
        assert report["train_gold_paths_added"] == 1
        assert report["selected_edges"] == 2
        assert report["metadata"]["metadata_entities_written"] == 3
        assert _read_rows(out_dir / "entity_types.txt") == [("D1", "0"), ("G1", "1"), ("E1", "2")]
        assert report["leakage_checks"]["non_train_evidence_paths_selected"] == 0
        assert report["leakage_checks"]["non_train_gold_paths_added"] == 0
        assert report["non_train_gold_paths_seen"] == 1
        assert report["leakage_checks"]["target_relation_edges_in_train1"] == 0
        assert report["gold_path_coverage"]["train"]["gold_path_coverage"] == 1.0
        assert report["gold_path_coverage"]["train"]["gold_edge_coverage"] == 1.0
        saved = json.loads((out_dir / "path_subgraph_report.json").read_text(encoding="utf-8"))
        assert saved["top_k_per_pair"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_path_subgraph_fills_gold_path_metadata_from_node_types():
    tmp_dir = _workspace_tmp_dir()
    try:
        source_dir = tmp_dir / "full"
        out_dir = tmp_dir / "subgraph"
        _write_tsv(source_dir / "train1.txt", [["D1", "r1", "E1"]])
        for name in ("train2.txt", "valid.txt", "test.txt"):
            _write_tsv(source_dir / name, [["D1", "affects_endpoint", "E1"]])
        _write_tsv(source_dir / "entity_names.txt", [["D1", "D1"], ["E1", "E1"]])
        _write_tsv(source_dir / "entity_types.txt", [["D1", 5], ["E1", 4]])
        (tmp_dir / "conversion_report.json").write_text(
            json.dumps({"entity_report": {"type_vocab": {"Drug": 5, "Disease": 4, "Protein": 14}}}),
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
                ["P1", "D1", "E1", "train", "p1", "D1|E1", "Drug|Endpoint", "r1", "1.0"],
            ],
        )
        _write_tsv(
            tmp_dir / "gold.tsv",
            [
                [
                    "pair_id",
                    "drug_id",
                    "endpoint_id",
                    "path_id",
                    "path_node_ids",
                    "path_node_types",
                    "path_relation_types",
                    "split",
                ],
                ["P1", "D1", "E1", "g1", "D1|PX|E1", "Drug|Protein|Endpoint", "rg|rh", "train"],
            ],
        )

        report = build_path_subgraph(
            {
                "source_dir": str(source_dir),
                "evidence_paths_train": str(tmp_dir / "evidence.tsv"),
                "gold_paths_file": str(tmp_dir / "gold.tsv"),
                "conversion_report": str(tmp_dir / "conversion_report.json"),
                "output_dir": str(out_dir),
                "top_k_per_pair": 1,
                "allowed_evidence_splits": ["train"],
            }
        )

        assert ("PX", "14") in _read_rows(out_dir / "entity_types.txt")
        assert ("PX", "PX") in _read_rows(out_dir / "entity_names.txt")
        assert report["metadata"]["missing_entity_names_filled_with_entity_id"] == 1
        assert report["metadata"]["missing_entity_types_filled_from_path_node_types"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
