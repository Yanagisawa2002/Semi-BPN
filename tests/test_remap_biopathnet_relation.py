import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.remap_biopathnet_relation import remap_biopathnet_relation


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"remap_bpn_relation_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _read_tsv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def test_remap_biopathnet_relation_changes_only_target_files():
    tmp_dir = _workspace_tmp_dir()
    try:
        source_dir = tmp_dir / "source"
        output_dir = tmp_dir / "out"
        _write_tsv(source_dir / "train1.txt", [["drug:D1", "drug_protein", "gene:G1"]])
        for name in ("train2.txt", "valid.txt", "test.txt"):
            _write_tsv(source_dir / name, [["drug:D1", "affects_endpoint", "disease:E1"]])
        _write_tsv(source_dir / "entity_types.txt", [["drug:D1", 1], ["gene:G1", 2], ["disease:E1", 3]])
        _write_tsv(source_dir / "entity_names.txt", [["drug:D1", "Drug 1"], ["gene:G1", "Gene 1"], ["disease:E1", "Endpoint 1"]])

        report = remap_biopathnet_relation(
            {
                "source_dir": str(source_dir),
                "output_dir": str(output_dir),
                "source_relation": "affects_endpoint",
                "target_relation": "indication",
            }
        )

        assert _read_tsv(output_dir / "train1.txt") == [["drug:D1", "drug_protein", "gene:G1"]]
        assert _read_tsv(output_dir / "train2.txt") == [["drug:D1", "indication", "disease:E1"]]
        assert _read_tsv(output_dir / "valid.txt") == [["drug:D1", "indication", "disease:E1"]]
        assert _read_tsv(output_dir / "test.txt") == [["drug:D1", "indication", "disease:E1"]]
        assert (output_dir / "entity_types.txt").exists()
        assert (output_dir / "entity_names.txt").exists()
        assert report["target_file_remapped_rows"] == 3
        assert report["leakage_checks"]["source_relation_rows_in_copied_files"] == 0
        assert report["leakage_checks"]["target_relation_rows_in_copied_files"] == 0

        saved = json.loads((output_dir / "relation_remap_report.json").read_text(encoding="utf-8"))
        assert saved["target_relation"] == "indication"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_remap_biopathnet_relation_rejects_mixed_target_relations_by_default():
    tmp_dir = _workspace_tmp_dir()
    try:
        source_dir = tmp_dir / "source"
        output_dir = tmp_dir / "out"
        _write_tsv(source_dir / "train1.txt", [["drug:D1", "drug_protein", "gene:G1"]])
        _write_tsv(source_dir / "train2.txt", [["drug:D1", "affects_endpoint", "disease:E1"]])
        _write_tsv(source_dir / "valid.txt", [["drug:D1", "already_other", "disease:E1"]])
        _write_tsv(source_dir / "test.txt", [["drug:D1", "affects_endpoint", "disease:E1"]])

        try:
            remap_biopathnet_relation(
                {
                    "source_dir": str(source_dir),
                    "output_dir": str(output_dir),
                    "source_relation": "affects_endpoint",
                    "target_relation": "indication",
                }
            )
        except ValueError as error:
            assert "relations other than 'affects_endpoint'" in str(error)
        else:
            raise AssertionError("Expected mixed target relations to be rejected")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_remap_biopathnet_relation_rejects_target_relation_in_copied_fact_graph():
    tmp_dir = _workspace_tmp_dir()
    try:
        source_dir = tmp_dir / "source"
        output_dir = tmp_dir / "out"
        _write_tsv(source_dir / "train1.txt", [["drug:D1", "indication", "disease:E1"]])
        for name in ("train2.txt", "valid.txt", "test.txt"):
            _write_tsv(source_dir / name, [["drug:D1", "affects_endpoint", "disease:E1"]])

        try:
            remap_biopathnet_relation(
                {
                    "source_dir": str(source_dir),
                    "output_dir": str(output_dir),
                    "source_relation": "affects_endpoint",
                    "target_relation": "indication",
                }
            )
        except ValueError as error:
            assert "target supervision edges must not appear" in str(error)
        else:
            raise AssertionError("Expected copied target relation rows to be rejected")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_k50_fallback_indication_prepare_config_remaps_only_supervision_files():
    config = yaml.safe_load(
        Path("configs/biopathnet_k50_fallback_indication_debug_prepare.yaml").read_text(encoding="utf-8")
    )

    assert config["source_dir"] == "data/cloud_run/biopathnet_path_subgraph_k50_fallback"
    assert config["output_dir"] == "data/cloud_run/biopathnet_path_subgraph_k50_fallback_indication_debug"
    assert config["source_relation"] == "affects_endpoint"
    assert config["target_relation"] == "indication"
    assert config["target_files"] == ["train2.txt", "valid.txt", "test.txt"]
    assert "train1.txt" not in config["target_files"]
    assert config["forbid_source_relation_in_copied_files"] is True
    assert config["forbid_target_relation_in_copied_files"] is True
