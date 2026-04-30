import csv
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.convert_all_data import (
    canonical_entity_id,
    linearize_graph,
    scan_kg_and_write_train1,
)


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"all_data_conversion_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_kg(path, rows):
    fieldnames = [
        "x_type",
        "x_source",
        "x_id",
        "x_name",
        "y_type",
        "y_source",
        "y_id",
        "y_name",
        "relation",
        "display_relation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_scan_kg_removes_prediction_edges_and_collects_bidirectional_pairs():
    tmp_dir = _workspace_tmp_dir()
    try:
        kg_path = tmp_dir / "kg.csv"
        train1_path = tmp_dir / "train1.txt"
        _write_kg(
            kg_path,
            [
                {
                    "x_type": "drug",
                    "x_source": "DrugBank",
                    "x_id": "DB1",
                    "x_name": "Drug 1",
                    "y_type": "disease",
                    "y_source": "MONDO",
                    "y_id": "E1",
                    "y_name": "Endpoint 1",
                    "relation": "indication",
                    "display_relation": "indication",
                },
                {
                    "x_type": "disease",
                    "x_source": "MONDO",
                    "x_id": "E2",
                    "x_name": "Endpoint 2",
                    "y_type": "drug",
                    "y_source": "DrugBank",
                    "y_id": "DB2",
                    "y_name": "Drug 2",
                    "relation": "indication",
                    "display_relation": "indication",
                },
                {
                    "x_type": "drug",
                    "x_source": "DrugBank",
                    "x_id": "DB1",
                    "x_name": "Drug 1",
                    "y_type": "gene/protein",
                    "y_source": "NCBI",
                    "y_id": "G1",
                    "y_name": "Gene 1",
                    "relation": "target",
                    "display_relation": "target",
                },
            ],
        )

        report = scan_kg_and_write_train1(kg_path, train1_path, remove_prediction_edges=True)

        expected_pairs = {
            (
                canonical_entity_id("drug", "DrugBank", "DB1"),
                canonical_entity_id("disease", "MONDO", "E1"),
            ),
            (
                canonical_entity_id("drug", "DrugBank", "DB2"),
                canonical_entity_id("disease", "MONDO", "E2"),
            ),
        }
        assert report["kg_positive_pairs"] == expected_pairs
        assert report["clinical_prediction_edges"] == 2

        lines = train1_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert "\ttarget\t" in lines[0]
        assert "indication" not in lines[0]
    finally:
        shutil.rmtree(tmp_dir)


def test_linearize_graph_rejects_non_linear_mechanism_graph():
    graph = {
        "nodes": [{"id": "D"}, {"id": "G1"}, {"id": "G2"}, {"id": "E"}],
        "links": [
            {"source": "D", "target": "G1", "key": "targets"},
            {"source": "D", "target": "G2", "key": "targets"},
            {"source": "G1", "target": "E", "key": "associated_with"},
        ],
    }

    node_order, reason = linearize_graph(graph)

    assert node_order is None
    assert reason == "non_linear_degree"
