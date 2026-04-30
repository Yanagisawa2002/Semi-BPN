"""Extract typed mechanism templates from curated gold mechanism paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

from mechrep.data.build_pairs import SPLITS, PairRecord, read_pair_tsv


REQUIRED_GOLD_PATH_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "path_id",
    "path_node_ids",
    "path_node_types",
    "path_relation_types",
    "split",
)

TEMPLATE_COLUMNS = (
    "template_id",
    "node_type_sequence",
    "relation_type_sequence",
    "support_count",
    "pair_ids",
    "path_ids",
)

EXTRACT_FROM_SPLITS = {
    "train_only": {"train"},
    "train_valid": {"train", "valid"},
}


@dataclass(frozen=True)
class GoldPathRecord:
    pair_id: str
    drug_id: str
    endpoint_id: str
    path_id: str
    node_ids: tuple[str, ...]
    node_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    split: str

    @property
    def node_type_sequence(self) -> str:
        return "|".join(self.node_types)

    @property
    def relation_type_sequence(self) -> str:
        return "|".join(self.relation_types)

    @property
    def template_key(self) -> str:
        return template_key(self.node_type_sequence, self.relation_type_sequence)


@dataclass(frozen=True)
class TemplateRecord:
    template_id: str
    node_type_sequence: str
    relation_type_sequence: str
    support_count: int
    pair_ids: tuple[str, ...]
    path_ids: tuple[str, ...]

    def as_row(self) -> dict:
        return {
            "template_id": self.template_id,
            "node_type_sequence": self.node_type_sequence,
            "relation_type_sequence": self.relation_type_sequence,
            "support_count": str(self.support_count),
            "pair_ids": "|".join(self.pair_ids),
            "path_ids": "|".join(self.path_ids),
        }


def template_key(node_type_sequence: str, relation_type_sequence: str) -> str:
    return f"{node_type_sequence} || {relation_type_sequence}"


def make_template_id(node_type_sequence: str, relation_type_sequence: str) -> str:
    digest = hashlib.sha1(template_key(node_type_sequence, relation_type_sequence).encode("utf-8")).hexdigest()[:12]
    return f"T_{digest}"


def split_sequence(value: str, *, sequence_separator: str, source: str, column: str) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        raise ValueError(f"{source} has empty {column}")
    parts = tuple(part.strip() for part in value.split(sequence_separator))
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"{source} has malformed {column}: empty sequence element")
    return parts


def gold_path_from_row(row: dict, *, sequence_separator: str, source: str) -> GoldPathRecord:
    missing_values = [column for column in REQUIRED_GOLD_PATH_COLUMNS if row.get(column) in (None, "")]
    if missing_values:
        raise ValueError(f"{source} has empty required columns: {missing_values}")

    split = row["split"]
    if split not in SPLITS:
        raise ValueError(f"{source} has unknown split {split!r}; expected one of {SPLITS}")

    node_ids = split_sequence(
        row["path_node_ids"],
        sequence_separator=sequence_separator,
        source=source,
        column="path_node_ids",
    )
    node_types = split_sequence(
        row["path_node_types"],
        sequence_separator=sequence_separator,
        source=source,
        column="path_node_types",
    )
    relation_types = split_sequence(
        row["path_relation_types"],
        sequence_separator=sequence_separator,
        source=source,
        column="path_relation_types",
    )

    if len(node_ids) < 2:
        raise ValueError(f"{source} has empty or too-short path_node_ids; at least two nodes are required")
    if len(node_types) != len(node_ids):
        raise ValueError(
            f"{source} has node_type length {len(node_types)} but node_id length {len(node_ids)}"
        )
    if len(relation_types) != len(node_types) - 1:
        raise ValueError(
            f"{source} has relation_type length {len(relation_types)} but expected node_type length - 1 "
            f"({len(node_types) - 1})"
        )

    return GoldPathRecord(
        pair_id=row["pair_id"],
        drug_id=row["drug_id"],
        endpoint_id=row["endpoint_id"],
        path_id=row["path_id"],
        node_ids=node_ids,
        node_types=node_types,
        relation_types=relation_types,
        split=split,
    )


def read_gold_paths(
    path: str | Path,
    *,
    delimiter: str = "\t",
    sequence_separator: str = "|",
) -> List[GoldPathRecord]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = [column for column in REQUIRED_GOLD_PATH_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        records = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has more fields than the header")
            records.append(
                gold_path_from_row(
                    row,
                    sequence_separator=sequence_separator,
                    source=f"{path}:{row_number}",
                )
            )

    if not records:
        raise ValueError(f"{path} contains no gold paths")
    return records


def load_pair_records(pair_split_dir: str | Path) -> Dict[str, Dict[str, PairRecord]]:
    pair_split_dir = Path(pair_split_dir)
    split_records = {}
    for split in SPLITS:
        split_path = pair_split_dir / f"{split}.tsv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing pair split file: {split_path}")
        records = read_pair_tsv(split_path, split=split)
        by_pair_id = {}
        for record in records:
            if record.pair_id in by_pair_id:
                raise ValueError(f"Duplicate pair_id {record.pair_id!r} in {split_path}")
            by_pair_id[record.pair_id] = record
        split_records[split] = by_pair_id
    return split_records


def validate_gold_paths_against_pair_splits(
    records: Sequence[GoldPathRecord],
    pair_records: Dict[str, Dict[str, PairRecord]],
) -> None:
    for record in records:
        split_pairs = pair_records[record.split]
        if record.pair_id not in split_pairs:
            raise ValueError(
                f"Gold path pair_id {record.pair_id!r} from split {record.split!r} "
                "is not found in the corresponding pair split file"
            )
        pair = split_pairs[record.pair_id]
        if pair.drug_id != record.drug_id or pair.endpoint_id != record.endpoint_id:
            raise ValueError(
                f"Gold path {record.path_id!r} pair metadata does not match pair split for "
                f"pair_id {record.pair_id!r}: gold has ({record.drug_id}, {record.endpoint_id}), "
                f"split has ({pair.drug_id}, {pair.endpoint_id})"
            )


def allowed_splits_for_extract_from(extract_from: str) -> set[str]:
    if extract_from not in EXTRACT_FROM_SPLITS:
        raise ValueError(f"Unsupported extract_from {extract_from!r}; expected one of {sorted(EXTRACT_FROM_SPLITS)}")
    return set(EXTRACT_FROM_SPLITS[extract_from])


def extract_templates(
    records: Sequence[GoldPathRecord],
    *,
    min_support: int = 1,
    extract_from: str = "train_only",
) -> List[TemplateRecord]:
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1, got {min_support}")

    allowed_splits = allowed_splits_for_extract_from(extract_from)
    selected = [record for record in records if record.split in allowed_splits]
    if not selected:
        raise ValueError(f"No gold paths available for template extraction with extract_from={extract_from!r}")

    groups = {}
    for record in selected:
        key = (record.node_type_sequence, record.relation_type_sequence)
        groups.setdefault(key, []).append(record)

    templates = []
    for (node_type_sequence, relation_type_sequence), group in groups.items():
        support_count = len(group)
        if support_count < min_support:
            continue
        templates.append(
            TemplateRecord(
                template_id=make_template_id(node_type_sequence, relation_type_sequence),
                node_type_sequence=node_type_sequence,
                relation_type_sequence=relation_type_sequence,
                support_count=support_count,
                pair_ids=tuple(sorted({record.pair_id for record in group})),
                path_ids=tuple(sorted({record.path_id for record in group})),
            )
        )

    templates.sort(key=lambda item: item.template_id)
    if not templates:
        raise ValueError(f"No templates passed min_support={min_support}")
    return templates


def write_templates(path: str | Path, templates: Sequence[TemplateRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for template in templates:
            writer.writerow(template.as_row())


def read_templates(path: str | Path) -> List[TemplateRecord]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = [column for column in TEMPLATE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        templates = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has more fields than the header")
            templates.append(
                TemplateRecord(
                    template_id=row["template_id"],
                    node_type_sequence=row["node_type_sequence"],
                    relation_type_sequence=row["relation_type_sequence"],
                    support_count=int(row["support_count"]),
                    pair_ids=tuple(filter(None, row["pair_ids"].split("|"))),
                    path_ids=tuple(filter(None, row["path_ids"].split("|"))),
                )
            )
    return templates


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping")
    return config


def create_toy_template_inputs(config: dict) -> None:
    gold_path = Path(config["gold_path_file"])
    pair_split_dir = Path(config["pair_split_dir"])
    pair_split_dir.mkdir(parents=True, exist_ok=True)
    for split, endpoint in [("train", "E_train"), ("valid", "E_valid"), ("test", "E_test")]:
        with (pair_split_dir / f"{split}.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["pair_id", "drug_id", "endpoint_id", "label"])
            writer.writerow([f"P_{split}_1", "D1", endpoint, "1"])
            writer.writerow([f"P_{split}_0", "D2", endpoint, "0"])

    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=config.get("delimiter", "\t"), lineterminator="\n")
        writer.writerow(REQUIRED_GOLD_PATH_COLUMNS)
        writer.writerow(
            [
                "P_train_1",
                "D1",
                "E_train",
                "path_train_1",
                "D1|G1|E_train",
                "Drug|Gene|Endpoint",
                "targets|associated_with",
                "train",
            ]
        )
        writer.writerow(
            [
                "P_valid_1",
                "D1",
                "E_valid",
                "path_valid_1",
                "D1|G2|E_valid",
                "Drug|Gene|Endpoint",
                "targets|associated_with",
                "valid",
            ]
        )
        writer.writerow(
            [
                "P_test_1",
                "D1",
                "E_test",
                "path_test_1",
                "D1|G3|E_test",
                "Drug|Gene|Endpoint",
                "targets|associated_with",
                "test",
            ]
        )


def run_template_extraction(config: dict, *, create_toy_data_if_missing: bool = False) -> List[TemplateRecord]:
    gold_path_file = Path(config["gold_path_file"])
    pair_split_dir = Path(config["pair_split_dir"])
    if create_toy_data_if_missing and (not gold_path_file.exists() or not pair_split_dir.exists()):
        create_toy_template_inputs(config)

    records = read_gold_paths(
        gold_path_file,
        delimiter=config.get("delimiter", "\t"),
        sequence_separator=config.get("sequence_separator", "|"),
    )
    pair_records = load_pair_records(pair_split_dir)
    validate_gold_paths_against_pair_splits(records, pair_records)
    templates = extract_templates(
        records,
        min_support=int(config.get("min_support", 1)),
        extract_from=config.get("extract_from", "train_only"),
    )
    output_dir = Path(config["output_dir"])
    write_templates(output_dir / "templates.tsv", templates)
    with (output_dir / "template_extraction_report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "extract_from": config.get("extract_from", "train_only"),
                "min_support": int(config.get("min_support", 1)),
                "num_gold_paths_total": len(records),
                "num_templates": len(templates),
                "test_paths_used_for_template_definition": 0,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/templates.yaml")
    parser.add_argument("--create-toy-data-if-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_template_extraction(config, create_toy_data_if_missing=args.create_toy_data_if_missing)


if __name__ == "__main__":
    main()
