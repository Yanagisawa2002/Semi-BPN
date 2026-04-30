"""Join endpoint pair splits with supervised gold-template labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from mechrep.data.build_pairs import PairRecord, SPLITS, load_pair_splits, validate_pair_ids_unique
from mechrep.templates.template_vocab import LABEL_TEMPLATE_COLUMNS, TemplateVocab


IGNORE_TEMPLATE_INDEX = -100


@dataclass(frozen=True)
class GoldTemplateExample:
    pair_id: str
    drug_id: str
    endpoint_id: str
    label: int
    split: str
    has_gold_template: bool
    primary_template_index: int
    template_ids: tuple[str, ...]
    primary_template_id: str
    num_gold_paths: int

    @classmethod
    def from_pair(
        cls,
        pair: PairRecord,
        *,
        template_label: "GoldTemplateLabelRow | None",
        template_vocab: TemplateVocab,
        unknown_template_policy: str = "error",
    ) -> "GoldTemplateExample":
        if pair.split is None:
            raise ValueError(f"Pair {pair.pair_id!r} is missing split metadata")

        if template_label is None:
            return cls(
                pair_id=pair.pair_id,
                drug_id=pair.drug_id,
                endpoint_id=pair.endpoint_id,
                label=pair.label,
                split=pair.split,
                has_gold_template=False,
                primary_template_index=IGNORE_TEMPLATE_INDEX,
                template_ids=(),
                primary_template_id="",
                num_gold_paths=0,
            )

        if template_label.drug_id != pair.drug_id or template_label.endpoint_id != pair.endpoint_id:
            raise ValueError(
                f"Gold-template label metadata does not match pair {pair.pair_id!r}: "
                f"label has ({template_label.drug_id}, {template_label.endpoint_id}), "
                f"pair has ({pair.drug_id}, {pair.endpoint_id})"
            )
        if template_label.split != pair.split:
            raise ValueError(
                f"Gold-template label for pair {pair.pair_id!r} has split {template_label.split!r}, "
                f"but pair split is {pair.split!r}"
            )

        missing = [
            template_id
            for template_id in set(template_label.template_ids + (template_label.primary_template_id,))
            if template_id not in template_vocab.template_id_to_index
        ]
        if missing and unknown_template_policy == "error":
            raise ValueError(
                f"Gold-template label for pair {pair.pair_id!r} references unknown template IDs: {sorted(missing)}"
            )
        if missing and unknown_template_policy != "ignore":
            raise ValueError(
                f"Unsupported unknown_template_policy {unknown_template_policy!r}; expected 'error' or 'ignore'"
            )

        has_known_primary = template_label.primary_template_id in template_vocab.template_id_to_index
        primary_index = (
            template_vocab.index(template_label.primary_template_id)
            if has_known_primary and pair.label == 1
            else IGNORE_TEMPLATE_INDEX
        )
        known_template_ids = tuple(
            template_id for template_id in template_label.template_ids if template_id in template_vocab.template_id_to_index
        )
        return cls(
            pair_id=pair.pair_id,
            drug_id=pair.drug_id,
            endpoint_id=pair.endpoint_id,
            label=pair.label,
            split=pair.split,
            has_gold_template=bool(known_template_ids and pair.label == 1 and has_known_primary),
            primary_template_index=primary_index,
            template_ids=known_template_ids,
            primary_template_id=template_label.primary_template_id if has_known_primary else "",
            num_gold_paths=template_label.num_gold_paths if known_template_ids else 0,
        )


@dataclass(frozen=True)
class GoldTemplateLabelRow:
    pair_id: str
    drug_id: str
    endpoint_id: str
    split: str
    template_ids: tuple[str, ...]
    primary_template_id: str
    num_gold_paths: int


def read_gold_template_label_tsv(path: str | Path) -> Dict[str, GoldTemplateLabelRow]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = [column for column in LABEL_TEMPLATE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        labels = {}
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has more fields than the header")
            pair_id = row["pair_id"]
            if not pair_id:
                raise ValueError(f"{path}:{row_number} has empty pair_id")
            if pair_id in labels:
                raise ValueError(f"{path} contains duplicate pair_id {pair_id!r}")
            template_ids = tuple(filter(None, row["template_ids"].split("|")))
            if not row["primary_template_id"]:
                raise ValueError(f"{path}:{row_number} has empty primary_template_id")
            labels[pair_id] = GoldTemplateLabelRow(
                pair_id=pair_id,
                drug_id=row["drug_id"],
                endpoint_id=row["endpoint_id"],
                split=row["split"],
                template_ids=template_ids,
                primary_template_id=row["primary_template_id"],
                num_gold_paths=int(row["num_gold_paths"]),
            )
        return labels


class GoldTemplatePairDataset:
    def __init__(self, examples_by_split: Dict[str, Sequence[GoldTemplateExample]]):
        missing = [split for split in SPLITS if split not in examples_by_split]
        if missing:
            raise ValueError(f"Missing dataset split(s): {missing}")
        self.examples_by_split = {split: list(examples_by_split[split]) for split in SPLITS}

    @classmethod
    def from_files(
        cls,
        *,
        split_dir: str | Path,
        train_labels: str | Path,
        valid_labels: str | Path,
        test_labels: str | Path,
        template_vocab: TemplateVocab,
        unknown_template_policy: str = "error",
    ) -> "GoldTemplatePairDataset":
        pair_splits = load_pair_splits(split_dir)
        validate_pair_ids_unique(example for records in pair_splits.values() for example in records)
        label_paths = {"train": train_labels, "valid": valid_labels, "test": test_labels}
        labels_by_split = {split: read_gold_template_label_tsv(path) for split, path in label_paths.items()}

        examples_by_split = {}
        for split in SPLITS:
            labels = labels_by_split[split]
            pair_ids = {pair.pair_id for pair in pair_splits[split]}
            extra_label_ids = sorted(set(labels) - pair_ids)
            if extra_label_ids:
                raise ValueError(
                    f"Gold-template labels for split {split!r} include pair_ids not present in pair split: "
                    f"{extra_label_ids[:10]}"
                )
            examples_by_split[split] = [
                GoldTemplateExample.from_pair(
                    pair,
                    template_label=labels.get(pair.pair_id),
                    template_vocab=template_vocab,
                    unknown_template_policy=unknown_template_policy,
                )
                for pair in pair_splits[split]
            ]
        return cls(examples_by_split)

    def examples(self, split: str) -> List[GoldTemplateExample]:
        if split not in self.examples_by_split:
            raise ValueError(f"Unknown split {split!r}")
        return list(self.examples_by_split[split])

    def all_examples(self) -> Iterable[GoldTemplateExample]:
        for split in SPLITS:
            yield from self.examples_by_split[split]
