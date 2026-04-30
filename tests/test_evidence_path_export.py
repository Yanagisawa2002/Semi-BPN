import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.templates.export_evidence_paths import export_evidence_paths


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"evidence_export_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _read_tsv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_export_evidence_paths_retrieves_train_no_gold_paths_only():
    tmp_dir = _workspace_tmp_dir()
    try:
        type_vocab = {
            "entity_report": {
                "type_vocab": {
                    "Drug": 0,
                    "Gene": 1,
                    "Endpoint": 2,
                }
            }
        }
        conversion_report = tmp_dir / "conversion_report.json"
        conversion_report.write_text(json.dumps(type_vocab), encoding="utf-8")
        _write_tsv(
            tmp_dir / "entity_types.txt",
            [
                ["D1", 0],
                ["D2", 0],
                ["G1", 1],
                ["G2", 1],
                ["E1", 2],
            ],
        )
        _write_tsv(
            tmp_dir / "train1.txt",
            [
                ["D1", "targets", "G1"],
                ["G1", "associated_with", "E1"],
                ["D2", "targets", "G2"],
                ["G2", "associated_with", "E1"],
            ],
        )
        _write_tsv(
            tmp_dir / "train.tsv",
            [
                ["pair_id", "drug_id", "endpoint_id", "label"],
                ["P_gold", "D1", "E1", "1"],
                ["P_no_gold", "D2", "E1", "1"],
                ["P_neg", "D1", "E1", "0"],
            ],
        )
        _write_tsv(
            tmp_dir / "gold.tsv",
            [
                ["pair_id", "drug_id", "endpoint_id", "split", "template_ids", "primary_template_id", "num_gold_paths"],
                ["P_gold", "D1", "E1", "train", "T1", "T1", "1"],
            ],
        )
        output_path = tmp_dir / "evidence_paths_train.tsv"
        report = export_evidence_paths(
            {
                "data": {
                    "kg_train": str(tmp_dir / "train1.txt"),
                    "train_pairs": str(tmp_dir / "train.tsv"),
                    "gold_labels_train": str(tmp_dir / "gold.tsv"),
                    "entity_types": str(tmp_dir / "entity_types.txt"),
                    "conversion_report": str(conversion_report),
                },
                "output": {"train_paths": str(output_path)},
                "retrieval": {"max_paths_per_pair": 10},
            }
        )

        rows = _read_tsv(output_path)
        assert report["target_train_positive_no_gold_pairs"] == 1
        assert len(rows) == 1
        assert rows[0]["pair_id"] == "P_no_gold"
        assert rows[0]["path_node_ids"] == "D2|G2|E1"
        assert rows[0]["path_node_types"] == "Drug|Gene|Endpoint"
        assert rows[0]["path_relation_types"] == "targets|associated_with"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_export_evidence_paths_guided_mechanism_path_with_relation_mapping():
    tmp_dir = _workspace_tmp_dir()
    try:
        type_vocab = {
            "entity_report": {
                "type_vocab": {
                    "Drug": 0,
                    "Gene": 1,
                    "BiologicalProcess": 2,
                    "Endpoint": 3,
                }
            }
        }
        conversion_report = tmp_dir / "conversion_report.json"
        conversion_report.write_text(json.dumps(type_vocab), encoding="utf-8")
        _write_tsv(
            tmp_dir / "entity_types.txt",
            [
                ["D1", 0],
                ["G1", 1],
                ["BP1", 2],
                ["G2", 1],
                ["E1", 3],
            ],
        )
        _write_tsv(
            tmp_dir / "train1.txt",
            [
                ["D1", "drug_protein", "G1"],
                ["G1", "bioprocess_protein", "BP1"],
                ["BP1", "bioprocess_protein", "G2"],
                ["G2", "disease_protein", "E1"],
            ],
        )
        _write_tsv(
            tmp_dir / "train.tsv",
            [
                ["pair_id", "drug_id", "endpoint_id", "label"],
                ["P_no_gold", "D1", "E1", "1"],
            ],
        )
        _write_tsv(
            tmp_dir / "gold.tsv",
            [
                ["pair_id", "drug_id", "endpoint_id", "split", "template_ids", "primary_template_id", "num_gold_paths"],
            ],
        )
        mapping_path = tmp_dir / "relation_mapping.yaml"
        mapping_path.write_text(
            "\n".join(
                [
                    "by_context:",
                    "  - source_type: Drug",
                    "    relation: drug_protein",
                    "    target_type: Gene",
                    "    mapped_relation: decreases_activity_of",
                    "  - source_type: Gene",
                    "    relation: bioprocess_protein",
                    "    target_type: BiologicalProcess",
                    "    mapped_relation: positively_regulates",
                    "  - source_type: BiologicalProcess",
                    "    relation: bioprocess_protein",
                    "    target_type: Gene",
                    "    mapped_relation: positively_regulates",
                    "  - source_type: Gene",
                    "    relation: disease_protein",
                    "    target_type: Endpoint",
                    "    mapped_relation: positively_correlated_with",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_path = tmp_dir / "evidence_paths_train.tsv"

        report = export_evidence_paths(
            {
                "data": {
                    "kg_train": str(tmp_dir / "train1.txt"),
                    "train_pairs": str(tmp_dir / "train.tsv"),
                    "gold_labels_train": str(tmp_dir / "gold.tsv"),
                    "entity_types": str(tmp_dir / "entity_types.txt"),
                    "conversion_report": str(conversion_report),
                },
                "output": {"train_paths": str(output_path)},
                "relation_mapping": {"enabled": True, "path": str(mapping_path)},
                "retrieval": {
                    "max_paths_per_pair": 10,
                    "max_guided_depth": 4,
                    "max_guided_first_hops_per_pair": 10,
                    "max_endpoint_predecessors_per_pair": 10,
                },
            }
        )

        rows = _read_tsv(output_path)
        mechanism_rows = [row for row in rows if row["path_node_types"] == "Drug|Gene|BiologicalProcess|Gene|Endpoint"]
        assert report["num_guided_first_hop_nodes"] == 1
        assert mechanism_rows
        assert (
            mechanism_rows[0]["path_relation_types"]
            == "decreases_activity_of|positively_regulates|positively_regulates|positively_correlated_with"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
