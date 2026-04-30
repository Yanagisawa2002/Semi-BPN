"""Utilities for endpoint-level drug-endpoint pair TSV files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PAIR_COLUMNS = ("pair_id", "drug_id", "endpoint_id", "label")
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    drug_id: str
    endpoint_id: str
    label: int
    split: str | None = None

    @classmethod
    def from_row(cls, row: dict, *, split: str | None = None, source: str = "<memory>") -> "PairRecord":
        missing = [column for column in PAIR_COLUMNS if row.get(column) in (None, "")]
        if missing:
            raise ValueError(f"{source} has empty required columns: {missing}")
        label = parse_label(row["label"], source=source)
        return cls(
            pair_id=str(row["pair_id"]),
            drug_id=str(row["drug_id"]),
            endpoint_id=str(row["endpoint_id"]),
            label=label,
            split=split,
        )

    def as_row(self, *, include_split: bool = False) -> dict:
        row = {
            "pair_id": self.pair_id,
            "drug_id": self.drug_id,
            "endpoint_id": self.endpoint_id,
            "label": str(self.label),
        }
        if include_split:
            row["split"] = self.split or ""
        return row


def parse_label(value: object, *, source: str = "<memory>") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} label must be 0 or 1, got boolean {value!r}")
    try:
        label = int(str(value))
    except ValueError as exc:
        raise ValueError(f"{source} label must be 0 or 1, got {value!r}") from exc
    if label not in (0, 1):
        raise ValueError(f"{source} label must be 0 or 1, got {value!r}")
    return label


def read_pair_tsv(path: str | Path, *, split: str | None = None) -> List[PairRecord]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")

        missing = [column for column in PAIR_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        records = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has more fields than the header")
            records.append(PairRecord.from_row(row, split=split, source=f"{path}:{row_number}"))

    if not records:
        raise ValueError(f"{path} contains no pair rows")
    return records


def write_pair_tsv(path: str | Path, records: Sequence[PairRecord], *, include_split: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(PAIR_COLUMNS) + (["split"] if include_split else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row(include_split=include_split))


def load_pair_splits(split_dir: str | Path) -> Dict[str, List[PairRecord]]:
    split_dir = Path(split_dir)
    splits = {}
    for split in SPLITS:
        path = split_dir / f"{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required split file: {path}")
        splits[split] = read_pair_tsv(path, split=split)
    return splits


def iter_records(splits: Dict[str, Sequence[PairRecord]]) -> Iterable[PairRecord]:
    for split in SPLITS:
        yield from splits[split]


def validate_pair_ids_unique(records: Iterable[PairRecord]) -> None:
    seen = set()
    duplicates = set()
    for record in records:
        if record.pair_id in seen:
            duplicates.add(record.pair_id)
        seen.add(record.pair_id)
    if duplicates:
        raise ValueError(f"Duplicate pair_id values found: {sorted(duplicates)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-tsv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_pair_tsv(args.input_tsv)
    validate_pair_ids_unique(records)
    write_pair_tsv(args.output_tsv, records)


if __name__ == "__main__":
    main()
