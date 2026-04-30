"""Dataset join for frozen pseudo-template semi-supervised training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from mechrep.data.build_pairs import PairRecord, SPLITS, load_pair_splits, validate_pair_ids_unique
from mechrep.data.gold_template_dataset import IGNORE_TEMPLATE_INDEX, GoldTemplateLabelRow, read_gold_template_label_tsv
from mechrep.templates.template_vocab import TemplateVocab


PSEUDO_TEMPLATE_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "pseudo_template_id",
    "pseudo_template_index",
    "matched_path_id",
    "pair_score",
    "template_match_score",
    "normalized_path_score",
    "final_confidence",
    "assignment_source",
)


@dataclass(frozen=True)
class PseudoTemplateLabelRow:
    pair_id: str
    drug_id: str
    endpoint_id: str
    split: str
    pseudo_template_id: str
    pseudo_template_index: int | None
    final_confidence: float
    assignment_source: str


@dataclass(frozen=True)
class SemiTemplateExample:
    pair_id: str
    drug_id: str
    endpoint_id: str
    label: int
    split: str
    has_gold_template: bool
    gold_template_id: str
    gold_template_index: int
    has_pseudo_template: bool
    pseudo_template_id: str
    pseudo_template_index: int
    template_supervision_source: str

    @classmethod
    def from_pair(
        cls,
        pair: PairRecord,
        *,
        gold_label: GoldTemplateLabelRow | None,
        pseudo_label: PseudoTemplateLabelRow | None,
        template_vocab: TemplateVocab,
        allow_gold_pseudo_overlap: bool = False,
        unknown_template_policy: str = "error",
    ) -> "SemiTemplateExample":
        if pair.split is None:
            raise ValueError(f"Pair {pair.pair_id!r} is missing split metadata")

        has_gold = False
        gold_template_id = ""
        gold_template_index = IGNORE_TEMPLATE_INDEX
        if gold_label is not None:
            _validate_label_metadata(pair, gold_label.pair_id, gold_label.drug_id, gold_label.endpoint_id, gold_label.split, "gold")
            unknown_gold = gold_label.primary_template_id not in template_vocab.template_id_to_index
            if unknown_gold and unknown_template_policy == "error":
                raise ValueError(
                    f"Gold-template label for pair {pair.pair_id!r} references unknown template ID "
                    f"{gold_label.primary_template_id!r}"
                )
            if unknown_gold and unknown_template_policy != "ignore":
                raise ValueError(
                    f"Unsupported unknown_template_policy {unknown_template_policy!r}; expected 'error' or 'ignore'"
                )
            if pair.label == 1 and not unknown_gold:
                has_gold = True
                gold_template_id = gold_label.primary_template_id
                gold_template_index = template_vocab.index(gold_template_id)

        has_pseudo = False
        pseudo_template_id = ""
        pseudo_template_index = IGNORE_TEMPLATE_INDEX
        if pseudo_label is not None:
            _validate_label_metadata(
                pair,
                pseudo_label.pair_id,
                pseudo_label.drug_id,
                pseudo_label.endpoint_id,
                pseudo_label.split,
                "pseudo",
            )
            if has_gold and not allow_gold_pseudo_overlap:
                raise ValueError(
                    f"Pair {pair.pair_id!r} has both gold-template and pseudo-template labels; "
                    "set allow_gold_pseudo_overlap=true to ignore pseudo supervision"
                )
            unknown_pseudo = pseudo_label.pseudo_template_id not in template_vocab.template_id_to_index
            if unknown_pseudo and unknown_template_policy == "error":
                raise ValueError(
                    f"Pseudo-template label for pair {pair.pair_id!r} references unknown template ID "
                    f"{pseudo_label.pseudo_template_id!r}"
                )
            if unknown_pseudo and unknown_template_policy != "ignore":
                raise ValueError(
                    f"Unsupported unknown_template_policy {unknown_template_policy!r}; expected 'error' or 'ignore'"
                )
            if pair.label == 1 and not has_gold and not unknown_pseudo:
                has_pseudo = True
                pseudo_template_id = pseudo_label.pseudo_template_id
                pseudo_template_index = template_vocab.index(pseudo_template_id)

        if has_gold:
            source = "gold"
        elif has_pseudo:
            source = "pseudo"
        else:
            source = "none"

        return cls(
            pair_id=pair.pair_id,
            drug_id=pair.drug_id,
            endpoint_id=pair.endpoint_id,
            label=pair.label,
            split=pair.split,
            has_gold_template=has_gold,
            gold_template_id=gold_template_id,
            gold_template_index=gold_template_index,
            has_pseudo_template=has_pseudo,
            pseudo_template_id=pseudo_template_id,
            pseudo_template_index=pseudo_template_index,
            template_supervision_source=source,
        )


def _validate_label_metadata(
    pair: PairRecord,
    label_pair_id: str,
    label_drug_id: str,
    label_endpoint_id: str,
    label_split: str,
    label_type: str,
) -> None:
    if label_pair_id != pair.pair_id:
        raise ValueError(f"{label_type} label pair_id mismatch for pair {pair.pair_id!r}")
    if label_drug_id != pair.drug_id or label_endpoint_id != pair.endpoint_id:
        raise ValueError(
            f"{label_type} label metadata does not match pair {pair.pair_id!r}: "
            f"label has ({label_drug_id}, {label_endpoint_id}), pair has ({pair.drug_id}, {pair.endpoint_id})"
        )
    if label_split != pair.split:
        raise ValueError(
            f"{label_type} label for pair {pair.pair_id!r} has split {label_split!r}, pair split is {pair.split!r}"
        )


def read_pseudo_template_label_tsv(path: str | Path) -> Dict[str, PseudoTemplateLabelRow]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = [column for column in PSEUDO_TEMPLATE_COLUMNS if column not in reader.fieldnames]
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
            pseudo_template_id = row["pseudo_template_id"]
            if not pseudo_template_id:
                raise ValueError(f"{path}:{row_number} has empty pseudo_template_id")
            pseudo_index = int(row["pseudo_template_index"]) if row["pseudo_template_index"] else None
            labels[pair_id] = PseudoTemplateLabelRow(
                pair_id=pair_id,
                drug_id=row["drug_id"],
                endpoint_id=row["endpoint_id"],
                split=row["split"],
                pseudo_template_id=pseudo_template_id,
                pseudo_template_index=pseudo_index,
                final_confidence=float(row["final_confidence"]),
                assignment_source=row["assignment_source"],
            )
        return labels


def cap_pseudo_labels_by_template(
    labels: Dict[str, PseudoTemplateLabelRow],
    *,
    max_per_template: int | None,
) -> Dict[str, PseudoTemplateLabelRow]:
    if max_per_template is None:
        return dict(labels)
    if max_per_template <= 0:
        raise ValueError(f"max_per_template must be positive when set, got {max_per_template}")

    labels_by_template: Dict[str, List[PseudoTemplateLabelRow]] = {}
    for label in labels.values():
        labels_by_template.setdefault(label.pseudo_template_id, []).append(label)

    capped = {}
    for template_id in sorted(labels_by_template):
        ranked = sorted(
            labels_by_template[template_id],
            key=lambda label: (-label.final_confidence, label.pair_id),
        )
        for label in ranked[:max_per_template]:
            capped[label.pair_id] = label
    return capped


class SemiTemplatePairDataset:
    def __init__(self, examples_by_split: Dict[str, Sequence[SemiTemplateExample]]):
        missing = [split for split in SPLITS if split not in examples_by_split]
        if missing:
            raise ValueError(f"Missing dataset split(s): {missing}")
        self.examples_by_split = {split: list(examples_by_split[split]) for split in SPLITS}

    @classmethod
    def from_files(
        cls,
        *,
        split_dir: str | Path,
        gold_labels_train: str | Path,
        gold_labels_valid: str | Path,
        gold_labels_test: str | Path,
        pseudo_labels_by_split: Dict[str, str | Path],
        template_vocab: TemplateVocab,
        use_pseudo_splits: Sequence[str] = ("train",),
        allow_gold_pseudo_overlap: bool = False,
        allow_test_pseudo: bool = False,
        unknown_template_policy: str = "error",
        max_pseudo_per_template: int | None = None,
    ) -> "SemiTemplatePairDataset":
        use_pseudo_splits = tuple(use_pseudo_splits)
        unknown_splits = sorted(set(use_pseudo_splits) - set(SPLITS))
        if unknown_splits:
            raise ValueError(f"Unknown use_pseudo_splits values: {unknown_splits}")
        if "test" in use_pseudo_splits and not allow_test_pseudo:
            raise ValueError("Test pseudo-template labels are rejected by default")

        pair_splits = load_pair_splits(split_dir)
        validate_pair_ids_unique(pair for records in pair_splits.values() for pair in records)
        gold_by_split = {
            "train": read_gold_template_label_tsv(gold_labels_train),
            "valid": read_gold_template_label_tsv(gold_labels_valid),
            "test": read_gold_template_label_tsv(gold_labels_test),
        }
        pseudo_by_split = {}
        for split in use_pseudo_splits:
            if split not in pseudo_labels_by_split:
                raise ValueError(f"Missing pseudo label path for split {split!r}")
            pseudo_by_split[split] = cap_pseudo_labels_by_template(
                read_pseudo_template_label_tsv(pseudo_labels_by_split[split]),
                max_per_template=max_pseudo_per_template,
            )

        examples_by_split = {}
        for split in SPLITS:
            pair_by_id = {pair.pair_id: pair for pair in pair_splits[split]}
            _validate_extra_ids(gold_by_split[split], pair_by_id, split, "gold-template")
            pseudo_labels = pseudo_by_split.get(split, {})
            _validate_extra_ids(pseudo_labels, pair_by_id, split, "pseudo-template")
            examples_by_split[split] = [
                SemiTemplateExample.from_pair(
                    pair,
                    gold_label=gold_by_split[split].get(pair.pair_id),
                    pseudo_label=pseudo_labels.get(pair.pair_id),
                    template_vocab=template_vocab,
                    allow_gold_pseudo_overlap=allow_gold_pseudo_overlap,
                    unknown_template_policy=unknown_template_policy,
                )
                for pair in pair_splits[split]
            ]
        return cls(examples_by_split)

    def examples(self, split: str) -> List[SemiTemplateExample]:
        if split not in self.examples_by_split:
            raise ValueError(f"Unknown split {split!r}")
        return list(self.examples_by_split[split])

    def all_examples(self) -> Iterable[SemiTemplateExample]:
        for split in SPLITS:
            yield from self.examples_by_split[split]


def _validate_extra_ids(labels: Dict[str, object], pair_by_id: Dict[str, PairRecord], split: str, label_type: str) -> None:
    extra = sorted(set(labels) - set(pair_by_id))
    if extra:
        raise ValueError(f"{label_type} labels for split {split!r} include unknown pair_ids: {extra[:10]}")
