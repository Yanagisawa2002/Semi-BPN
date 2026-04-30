import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.no_evidence_report import build_no_evidence_report


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"no_evidence_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def test_no_evidence_report_splits_gold_exclusions_from_retrieval_misses():
    tmp_dir = _workspace_tmp_dir()
    try:
        _write_tsv(
            tmp_dir / "train.tsv",
            [
                ["pair_id", "drug_id", "endpoint_id", "label"],
                ["P1", "D1", "E1", "1"],
                ["P2", "D2", "E2", "1"],
                ["P3", "D3", "E3", "0"],
                ["P4", "D4", "E4", "1"],
            ],
        )
        _write_tsv(
            tmp_dir / "train1.txt",
            [
                ["D1", "drug_protein", "G1"],
                ["G1", "disease_protein", "E1"],
                ["D2", "drug_protein", "G2"],
                ["G2", "disease_protein", "E2"],
                ["D4", "drug_protein", "G4"],
                ["G4", "disease_protein", "E4"],
            ],
        )
        _write_tsv(
            tmp_dir / "entity_types.txt",
            [["D1", 0], ["D2", 0], ["D4", 0], ["G1", 1], ["G2", 1], ["G4", 1], ["E1", 2], ["E2", 2], ["E4", 2]],
        )
        (tmp_dir / "conversion_report.json").write_text(
            json.dumps({"entity_report": {"type_vocab": {"Drug": 0, "Gene": 1, "Disease": 2}}}),
            encoding="utf-8",
        )
        _write_tsv(
            tmp_dir / "gold.tsv",
            [
                ["pair_id", "drug_id", "endpoint_id", "split", "template_ids", "primary_template_id", "num_gold_paths"],
                ["P4", "D4", "E4", "train", "T1", "T1", "1"],
            ],
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
                ["P2", "D2", "E2", "train", "EV1", "D2|G2|E2", "Drug|Gene|Endpoint", "drug_protein|disease_protein", "1.0"],
            ],
        )

        summary = build_no_evidence_report(
            {
                "data": {
                    "train_pairs": str(tmp_dir / "train.tsv"),
                    "kg_train": str(tmp_dir / "train1.txt"),
                    "entity_types": str(tmp_dir / "entity_types.txt"),
                    "conversion_report": str(tmp_dir / "conversion_report.json"),
                    "gold_labels_train": str(tmp_dir / "gold.tsv"),
                    "evidence_paths_train": str(tmp_dir / "evidence.tsv"),
                },
                "output": {
                    "no_evidence_pairs": str(tmp_dir / "reports" / "no_evidence.tsv"),
                    "summary_json": str(tmp_dir / "reports" / "summary.json"),
                },
            }
        )

        assert summary["train_positive_pairs"] == 3
        assert summary["train_positive_pairs_with_evidence_paths"] == 1
        assert summary["train_positive_pairs_without_evidence_paths"] == 2
        assert summary["no_evidence_gold_template_pairs"] == 1
        assert summary["no_evidence_no_gold_positive_pairs"] == 1
        assert summary["leakage_checks"]["reported_negative_pairs"] == 0
        with (tmp_dir / "reports" / "no_evidence.tsv").open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert {row["pair_id"] for row in rows} == {"P1", "P4"}
        assert {row["no_evidence_scope"] for row in rows} == {
            "gold_positive_already_supervised",
            "no_gold_positive_retrieval_miss",
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
