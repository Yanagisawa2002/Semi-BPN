import csv
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.compare_graph_variants import compare_graph_variants


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"graph_compare_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def test_graph_variant_comparison_writes_deterministic_summary():
    tmp_dir = _workspace_tmp_dir()
    try:
        full = tmp_dir / "full"
        sub = tmp_dir / "sub"
        _write_tsv(full / "train1.txt", [["A", "r", "B"], ["B", "r", "C"]])
        _write_tsv(sub / "train1.txt", [["A", "r", "B"]])
        payload = compare_graph_variants(
            {
                "variants": [
                    {"name": "full", "graph_dir": str(full), "smoke_command": "full.sh"},
                    {"name": "sub", "graph_dir": str(sub), "smoke_command": "sub.sh"},
                ],
                "output": {
                    "summary_tsv": str(tmp_dir / "reports" / "compare.tsv"),
                    "summary_json": str(tmp_dir / "reports" / "compare.json"),
                },
            }
        )

        assert [row["variant"] for row in payload["variants"]] == ["full", "sub"]
        assert payload["variants"][0]["edge_reduction_vs_full"] == 1.0
        assert payload["variants"][1]["edge_reduction_vs_full"] == 2.0
        assert (tmp_dir / "reports" / "compare.tsv").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
