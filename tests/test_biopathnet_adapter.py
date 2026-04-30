import csv
import hashlib
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.biopathnet_adapter import EndpointBioPathNetAdapter


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"adapter_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_split(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["pair_id", "drug_id", "endpoint_id", "label"])
        writer.writerows(rows)


def _write_toy_splits(split_dir):
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_split(
        split_dir / "train.tsv",
        [
            ("p_train_0", "drug_a", "endpoint_a", "1"),
            ("p_train_1", "drug_b", "endpoint_a", "0"),
        ],
    )
    _write_split(
        split_dir / "valid.tsv",
        [
            ("p_valid_0", "drug_a", "endpoint_b", "1"),
            ("p_valid_1", "drug_b", "endpoint_b", "0"),
        ],
    )
    _write_split(
        split_dir / "test.tsv",
        [
            ("p_test_0", "drug_a", "endpoint_c", "1"),
            ("p_test_1", "drug_b", "endpoint_c", "0"),
        ],
    )


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_adapter_preserves_pair_metadata_and_maps_to_triples():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir = tmp_dir / "splits"
        _write_toy_splits(split_dir)

        adapter = EndpointBioPathNetAdapter.from_split_dir(split_dir, relation_name="affects_endpoint")
        triples = adapter.triples("test")

        assert [(triple.pair_id, triple.drug_id, triple.endpoint_id, triple.label) for triple in triples] == [
            ("p_test_0", "drug_a", "endpoint_c", 1),
            ("p_test_1", "drug_b", "endpoint_c", 0),
        ]
        assert [triple.as_biopathnet_row() for triple in triples] == [
            ("drug_a", "affects_endpoint", "endpoint_c"),
            ("drug_b", "affects_endpoint", "endpoint_c"),
        ]
        assert [triple.as_biopathnet_row() for triple in adapter.triples("test", positive_only=True)] == [
            ("drug_a", "affects_endpoint", "endpoint_c")
        ]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_adapter_output_is_deterministic_and_does_not_touch_split_files():
    tmp_dir = _workspace_tmp_dir()
    try:
        split_dir = tmp_dir / "splits"
        _write_toy_splits(split_dir)
        before = {path.name: _digest(path) for path in sorted(split_dir.glob("*.tsv"))}

        adapter = EndpointBioPathNetAdapter.from_split_dir(split_dir)
        output_a = tmp_dir / "adapter_a"
        output_b = tmp_dir / "adapter_b"
        counts_a = adapter.write_adapter_outputs(output_a)
        counts_b = adapter.write_adapter_outputs(output_b)

        after = {path.name: _digest(path) for path in sorted(split_dir.glob("*.tsv"))}
        assert before == after
        assert counts_a == counts_b == {"train": 1, "valid": 1, "test": 1}

        files = [
            "train_metadata.tsv",
            "valid_metadata.tsv",
            "test_metadata.tsv",
            "biopathnet_triples/train2.txt",
            "biopathnet_triples/valid.txt",
            "biopathnet_triples/test.txt",
        ]
        for file_name in files:
            assert (output_a / file_name).read_text(encoding="utf-8") == (
                output_b / file_name
            ).read_text(encoding="utf-8")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
