"""Adapter from endpoint pair splits to BioPathNet-style triples."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from mechrep.data.build_pairs import PairRecord, SPLITS, iter_records, load_pair_splits, validate_pair_ids_unique


DEFAULT_RELATION_NAME = "affects_endpoint"


@dataclass(frozen=True)
class BioPathNetTriple:
    head: str
    relation: str
    tail: str
    pair_id: str
    drug_id: str
    endpoint_id: str
    label: int
    split: str

    def as_biopathnet_row(self) -> tuple[str, str, str]:
        return self.head, self.relation, self.tail

    def as_metadata_row(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "drug_id": self.drug_id,
            "endpoint_id": self.endpoint_id,
            "label": str(self.label),
            "split": self.split,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
        }


class EndpointBioPathNetAdapter:
    """Convert endpoint pair splits into triples while preserving pair metadata."""

    metadata_columns = ("pair_id", "drug_id", "endpoint_id", "label", "split", "head", "relation", "tail")

    def __init__(self, splits: Dict[str, Sequence[PairRecord]], relation_name: str = DEFAULT_RELATION_NAME):
        if not relation_name:
            raise ValueError("relation_name must be non-empty")
        for split in SPLITS:
            if split not in splits:
                raise ValueError(f"Missing split {split!r}")
        validate_pair_ids_unique(iter_records(splits))
        self.splits = {split: list(records) for split, records in splits.items()}
        self.relation_name = relation_name

    @classmethod
    def from_split_dir(
        cls,
        split_dir: str | Path,
        *,
        relation_name: str = DEFAULT_RELATION_NAME,
    ) -> "EndpointBioPathNetAdapter":
        return cls(load_pair_splits(split_dir), relation_name=relation_name)

    def triples(self, split: str | None = None, *, positive_only: bool = False) -> List[BioPathNetTriple]:
        split_names = [split] if split is not None else list(SPLITS)
        triples = []
        for split_name in split_names:
            if split_name not in self.splits:
                raise ValueError(f"Unknown split {split_name!r}")
            for record in self.splits[split_name]:
                if positive_only and record.label != 1:
                    continue
                triples.append(
                    BioPathNetTriple(
                        head=record.drug_id,
                        relation=self.relation_name,
                        tail=record.endpoint_id,
                        pair_id=record.pair_id,
                        drug_id=record.drug_id,
                        endpoint_id=record.endpoint_id,
                        label=record.label,
                        split=split_name,
                    )
                )
        return triples

    def records(self, split: str) -> List[PairRecord]:
        if split not in self.splits:
            raise ValueError(f"Unknown split {split!r}")
        return list(self.splits[split])

    def drug_vocab(self) -> List[str]:
        return sorted({record.drug_id for record in iter_records(self.splits)})

    def endpoint_vocab(self) -> List[str]:
        return sorted({record.endpoint_id for record in iter_records(self.splits)})

    def write_metadata(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            path = output_dir / f"{split}_metadata.tsv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self.metadata_columns,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                for triple in self.triples(split):
                    writer.writerow(triple.as_metadata_row())

    def write_biopathnet_triples(self, output_dir: str | Path, *, positive_only: bool = True) -> dict:
        """Write triple files compatible with BioPathNet's h/r/t TSV reader.

        BioPathNet's KGC task treats every row as a positive triple and samples
        negatives internally. Negative-labeled endpoint pairs are therefore not
        written to these triple files by default, but they remain preserved in
        the metadata files and in binary baseline evaluation.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_names = {"train": "train2.txt", "valid": "valid.txt", "test": "test.txt"}
        counts = {}
        for split, file_name in file_names.items():
            triples = self.triples(split, positive_only=positive_only)
            path = output_dir / file_name
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                for triple in triples:
                    writer.writerow(triple.as_biopathnet_row())
            counts[split] = len(triples)
        return counts

    def write_adapter_outputs(self, output_dir: str | Path) -> dict:
        output_dir = Path(output_dir)
        self.write_metadata(output_dir)
        return self.write_biopathnet_triples(output_dir / "biopathnet_triples", positive_only=True)
