"""Build endpoint-level OOD splits for drug-endpoint pair prediction."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REQUIRED_COLUMNS = ("pair_id", "drug_id", "endpoint_id", "label")


def read_pair_tsv(path: str | Path) -> tuple[List[dict], List[str]]:
    """Read a pair TSV and validate the required schema."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")

        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has more fields than the header")
            for column in REQUIRED_COLUMNS:
                if row.get(column) in (None, ""):
                    raise ValueError(f"{path}:{row_number} has empty required column {column!r}")
            rows.append(row)

    if not rows:
        raise ValueError(f"{path} contains no pair rows")

    return rows, list(reader.fieldnames)


def write_pair_tsv(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    """Write rows to a TSV file using the original input column order."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _count_from_ratio(num_items: int, ratio: float, name: str) -> int:
    if ratio < 0 or ratio >= 1:
        raise ValueError(f"{name} must be in [0, 1), got {ratio}")
    if ratio == 0:
        return 0
    return max(1, round(num_items * ratio))


def _endpoint_sets(
    endpoints: Sequence[str],
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be less than 1")

    shuffled = sorted(set(endpoints))
    if len(shuffled) < 3 and valid_ratio > 0 and test_ratio > 0:
        raise ValueError("endpoint_ood split needs at least 3 endpoints when valid and test are non-empty")

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    num_endpoints = len(shuffled)
    num_test = _count_from_ratio(num_endpoints, test_ratio, "test_ratio")
    num_valid = _count_from_ratio(num_endpoints, valid_ratio, "valid_ratio")

    while num_valid + num_test >= num_endpoints:
        if num_valid >= num_test and num_valid > 0:
            num_valid -= 1
        elif num_test > 0:
            num_test -= 1
        else:
            break

    test_endpoints = set(shuffled[:num_test])
    valid_endpoints = set(shuffled[num_test : num_test + num_valid])
    train_endpoints = set(shuffled[num_test + num_valid :])

    if not train_endpoints:
        raise ValueError("endpoint_ood split produced no training endpoints")
    if test_ratio > 0 and not test_endpoints:
        raise ValueError("endpoint_ood split produced no test endpoints")
    if valid_ratio > 0 and not valid_endpoints:
        raise ValueError("endpoint_ood split produced no validation endpoints")

    return train_endpoints, valid_endpoints, test_endpoints


def split_endpoint_ood(
    rows: Sequence[dict],
    *,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> Dict[str, List[dict]]:
    """Split rows so each endpoint appears in exactly one split."""
    endpoints = [row["endpoint_id"] for row in rows]
    train_endpoints, valid_endpoints, test_endpoints = _endpoint_sets(
        endpoints,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    splits = {"train": [], "valid": [], "test": []}
    for row in rows:
        endpoint = row["endpoint_id"]
        if endpoint in train_endpoints:
            splits["train"].append(row)
        elif endpoint in valid_endpoints:
            splits["valid"].append(row)
        elif endpoint in test_endpoints:
            splits["test"].append(row)
        else:
            raise RuntimeError(f"Endpoint {endpoint!r} was not assigned to any split")

    return splits


def _endpoints(rows: Iterable[dict]) -> set[str]:
    return {row["endpoint_id"] for row in rows}


def build_leakage_report(splits: Dict[str, Sequence[dict]], *, seed: int, split_type: str) -> dict:
    train_endpoints = _endpoints(splits["train"])
    valid_endpoints = _endpoints(splits["valid"])
    test_endpoints = _endpoints(splits["test"])

    train_test_overlap = train_endpoints & test_endpoints
    train_valid_overlap = train_endpoints & valid_endpoints
    valid_test_overlap = valid_endpoints & test_endpoints

    return {
        "split_type": split_type,
        "seed": seed,
        "number_of_train_endpoints": len(train_endpoints),
        "number_of_valid_endpoints": len(valid_endpoints),
        "number_of_test_endpoints": len(test_endpoints),
        "train_rows": len(splits["train"]),
        "valid_rows": len(splits["valid"]),
        "test_rows": len(splits["test"]),
        "train_test_endpoint_overlap": len(train_test_overlap),
        "train_valid_endpoint_overlap": len(train_valid_overlap),
        "valid_test_endpoint_overlap": len(valid_test_overlap),
        "train_test_endpoint_overlap_values": sorted(train_test_overlap),
        "train_valid_endpoint_overlap_values": sorted(train_valid_overlap),
        "valid_test_endpoint_overlap_values": sorted(valid_test_overlap),
    }


def build_ood_splits(
    input_tsv: str | Path,
    output_dir: str | Path,
    *,
    split_type: str = "endpoint_ood",
    seed: int = 0,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> dict:
    """Build and save endpoint-OOD split files plus a leakage report."""
    if split_type != "endpoint_ood":
        raise ValueError(f"Unsupported split_type {split_type!r}; expected 'endpoint_ood'")

    rows, fieldnames = read_pair_tsv(input_tsv)
    splits = split_endpoint_ood(rows, valid_ratio=valid_ratio, test_ratio=test_ratio, seed=seed)
    report = build_leakage_report(splits, seed=seed, split_type=split_type)

    if report["train_test_endpoint_overlap"] != 0:
        raise RuntimeError("Endpoint leakage detected between train and test")
    if report["train_valid_endpoint_overlap"] != 0:
        raise RuntimeError("Endpoint leakage detected between train and validation")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in splits.items():
        write_pair_tsv(output_dir / f"{split_name}.tsv", split_rows, fieldnames)

    with (output_dir / "leakage_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", required=True, help="Input TSV with pair_id, drug_id, endpoint_id, label")
    parser.add_argument("--output-dir", required=True, help="Directory for train.tsv, valid.tsv, test.tsv")
    parser.add_argument("--split-type", default="endpoint_ood", choices=["endpoint_ood"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_ood_splits(
        args.input_tsv,
        args.output_dir,
        split_type=args.split_type,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
