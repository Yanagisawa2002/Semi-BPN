import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.prepare_biopathnet_full import prepare_biopathnet_full


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"prepare_bpn_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def test_prepare_biopathnet_full_adds_missing_disease_metadata():
    tmp_dir = _workspace_tmp_dir()
    try:
        source_dir = tmp_dir / "source"
        output_dir = tmp_dir / "out"
        for name in ("train1.txt", "train2.txt", "valid.txt", "test.txt"):
            _write_tsv(source_dir / name, [["drug:D1", "affects_endpoint", "disease:E_missing"]])
        _write_tsv(source_dir / "entity_types.txt", [["drug:D1", 5]])
        _write_tsv(source_dir / "entity_names.txt", [["drug:D1", "Drug 1"]])
        report_path = tmp_dir / "conversion_report.json"
        report_path.write_text(
            json.dumps({"entity_report": {"type_vocab": {"Disease": 4, "Drug": 5, "Gene": 7}}}),
            encoding="utf-8",
        )

        report = prepare_biopathnet_full(
            {
                "source_dir": str(source_dir),
                "output_dir": str(output_dir),
                "conversion_report": str(report_path),
            }
        )

        assert report["missing_type_entities_added"] == 1
        assert report["missing_name_entities_added"] == 1
        assert (output_dir / "train1.txt").exists()
        types = dict(csv.reader((output_dir / "entity_types.txt").open(encoding="utf-8"), delimiter="\t"))
        names = dict(csv.reader((output_dir / "entity_names.txt").open(encoding="utf-8"), delimiter="\t"))
        assert types["disease:E_missing"] == "4"
        assert names["disease:E_missing"] == "disease:E_missing"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
