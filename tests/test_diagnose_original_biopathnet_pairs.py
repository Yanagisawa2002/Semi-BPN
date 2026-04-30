import shutil
from pathlib import Path

from mechrep.data.build_pairs import PairRecord
from mechrep.evaluation.diagnose_original_biopathnet_pairs import (
    endpoint_overlap_report,
    factgraph_support_report,
    limit_records,
    limit_triplets,
    read_triplets,
    relation_counts_for_files,
    summarize_pair_records,
    summarize_scores,
)


def test_summarize_scores_reports_positive_negative_gap():
    rows = [
        {"pair_id": "p1", "label": "1", "score": "0.9"},
        {"pair_id": "p2", "label": "1", "score": "0.7"},
        {"pair_id": "n1", "label": "0", "score": "0.2"},
        {"pair_id": "n2", "label": "0", "score": "0.4"},
    ]

    summary = summarize_scores(rows)

    assert summary["positive"]["count"] == 2
    assert summary["negative"]["count"] == 2
    assert summary["mean_positive_minus_negative"] == 0.5
    assert summary["positive"]["median"] == 0.8


def test_limit_records_is_deterministic_and_label_balanced():
    records = [
        PairRecord(pair_id=f"p{i}", drug_id=f"d{i}", endpoint_id="e", label=1)
        for i in range(10)
    ] + [
        PairRecord(pair_id=f"n{i}", drug_id=f"d{i}", endpoint_id="e", label=0)
        for i in range(10)
    ]

    left = limit_records(records, max_records=6, seed=7)
    right = limit_records(records, max_records=6, seed=7)

    assert left == right
    assert len(left) == 6
    assert sum(record.label for record in left) == 3


def test_read_and_limit_triplets_are_deterministic():
    dataset_dir = Path("results/test_tmp/diagnose_original_biopathnet_pairs/triplets")
    shutil.rmtree(dataset_dir.parent, ignore_errors=True)
    dataset_dir.mkdir(parents=True)
    path = dataset_dir / "train2.txt"
    path.write_text(
        "\n".join(
            [
                "d1\tr\te1",
                "d2\tr\te2",
                "d3\tr\te3",
                "d4\tr\te4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    triplets = read_triplets(path)
    limited_left = limit_triplets(triplets, max_triplets=2, seed=5)
    limited_right = limit_triplets(triplets, max_triplets=2, seed=5)

    assert triplets[0] == ("d1", "r", "e1")
    assert limited_left == limited_right
    assert len(limited_left) == 2

    shutil.rmtree(dataset_dir.parent, ignore_errors=True)


def test_endpoint_overlap_report_flags_train_test_overlap():
    records = {
        "train": [PairRecord("a", "d1", "e_shared", 1), PairRecord("b", "d2", "e_train", 0)],
        "test": [PairRecord("c", "d1", "e_shared", 1), PairRecord("d", "d3", "e_test", 0)],
    }

    report = endpoint_overlap_report(records)

    assert report["endpoint_ood_leakage_flags"]["train_test_endpoint_overlap"] == 1
    assert report["drugs"]["pairwise_overlap"]["test_train"]["count"] == 1


def test_relation_counts_and_factgraph_support():
    dataset_dir = Path("results/test_tmp/diagnose_original_biopathnet_pairs/dataset")
    shutil.rmtree(dataset_dir.parent, ignore_errors=True)
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train2.txt").write_text("d1\taffects_endpoint\te1\n", encoding="utf-8")
    (dataset_dir / "valid.txt").write_text("d2\taffects_endpoint\te2\n", encoding="utf-8")
    (dataset_dir / "test.txt").write_text("d3\tother\te3\n", encoding="utf-8")
    (dataset_dir / "train1.txt").write_text(
        "\n".join(
            [
                "d1\tr\tx",
                "x\tr\te1",
                "d2\tr\te2",
                "e3\tr\tx",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = {
        "train": [PairRecord("a", "d1", "e1", 1)],
        "valid": [PairRecord("b", "d2", "e2", 1)],
        "test": [PairRecord("c", "d3", "e3", 1)],
    }

    relation_report = relation_counts_for_files(dataset_dir, ["train2.txt", "valid.txt", "test.txt"])
    support = factgraph_support_report(dataset_dir, records)

    assert relation_report["train2.txt"]["relation_counts"] == {"affects_endpoint": 1}
    assert relation_report["test.txt"]["relation_counts"] == {"other": 1}
    assert support["splits"]["train"]["num_endpoints_with_in_edges"] == 1
    assert support["splits"]["test"]["num_endpoints_with_any_incident_edge"] == 1

    shutil.rmtree(dataset_dir.parent, ignore_errors=True)


def test_summarize_pair_records_counts_label_balance():
    records = [
        PairRecord("a", "d1", "e1", 1),
        PairRecord("b", "d1", "e1", 0),
        PairRecord("c", "d2", "e2", 1),
    ]

    summary = summarize_pair_records(records)

    assert summary["num_pairs"] == 3
    assert summary["num_positive"] == 2
    assert summary["num_negative"] == 1
    assert summary["num_unique_endpoints"] == 2
