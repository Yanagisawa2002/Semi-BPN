"""Build gold-template labels for endpoint pairs from curated mechanism paths."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from mechrep.templates.extract_templates import (
    TEMPLATE_COLUMNS,
    GoldPathRecord,
    TemplateRecord,
    allowed_splits_for_extract_from,
    load_config,
    load_pair_records,
    make_template_id,
    read_gold_paths,
    read_templates,
    run_template_extraction,
    template_key,
    validate_gold_paths_against_pair_splits,
)


LABEL_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "template_ids",
    "primary_template_id",
    "num_gold_paths",
)


@dataclass(frozen=True)
class GoldTemplateLabel:
    pair_id: str
    drug_id: str
    endpoint_id: str
    split: str
    template_ids: tuple[str, ...]
    primary_template_id: str
    num_gold_paths: int

    def as_row(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "drug_id": self.drug_id,
            "endpoint_id": self.endpoint_id,
            "split": self.split,
            "template_ids": "|".join(self.template_ids),
            "primary_template_id": self.primary_template_id,
            "num_gold_paths": str(self.num_gold_paths),
        }


def template_lookup(templates: Sequence[TemplateRecord]) -> Dict[str, TemplateRecord]:
    lookup = {}
    for template in templates:
        key = template_key(template.node_type_sequence, template.relation_type_sequence)
        if key in lookup:
            raise ValueError(f"Duplicate template key in template table: {key}")
        lookup[key] = template
    return lookup


def select_primary_template(template_ids: Sequence[str], support_by_template_id: Dict[str, int]) -> str:
    if not template_ids:
        raise ValueError("Cannot select a primary template from an empty template list")
    unique_template_ids = sorted(set(template_ids))
    return sorted(unique_template_ids, key=lambda template_id: (-support_by_template_id[template_id], template_id))[0]


def build_gold_template_labels(
    records: Sequence[GoldPathRecord],
    templates: Sequence[TemplateRecord],
) -> List[GoldTemplateLabel]:
    lookup = template_lookup(templates)
    support_by_template_id = {template.template_id: template.support_count for template in templates}
    grouped = {}
    for record in records:
        key = record.template_key
        if key not in lookup:
            continue
        grouped.setdefault(record.pair_id, {"record": record, "template_ids": [], "path_ids": []})
        grouped[record.pair_id]["template_ids"].append(lookup[key].template_id)
        grouped[record.pair_id]["path_ids"].append(record.path_id)

    labels = []
    split_order = {"train": 0, "valid": 1, "test": 2}
    ordered_pair_ids = sorted(
        grouped,
        key=lambda pair_id: (split_order[grouped[pair_id]["record"].split], pair_id),
    )
    for pair_id in ordered_pair_ids:
        group = grouped[pair_id]
        record = group["record"]
        template_ids = tuple(sorted(set(group["template_ids"])))
        labels.append(
            GoldTemplateLabel(
                pair_id=pair_id,
                drug_id=record.drug_id,
                endpoint_id=record.endpoint_id,
                split=record.split,
                template_ids=template_ids,
                primary_template_id=select_primary_template(template_ids, support_by_template_id),
                num_gold_paths=len(group["path_ids"]),
            )
        )
    return labels


def write_labels(path: str | Path, labels: Sequence[GoldTemplateLabel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for label in labels:
            writer.writerow(label.as_row())


def write_split_labels(output_dir: str | Path, labels: Sequence[GoldTemplateLabel]) -> None:
    output_dir = Path(output_dir)
    write_labels(output_dir / "gold_template_labels.tsv", labels)
    for split in ("train", "valid", "test"):
        write_labels(
            output_dir / f"gold_template_labels_{split}.tsv",
            [label for label in labels if label.split == split],
        )


def run_gold_template_label_generation(config: dict, *, create_toy_data_if_missing: bool = False) -> List[GoldTemplateLabel]:
    output_dir = Path(config["output_dir"])
    templates_path = output_dir / "templates.tsv"
    if not templates_path.exists():
        run_template_extraction(config, create_toy_data_if_missing=create_toy_data_if_missing)

    records = read_gold_paths(
        config["gold_path_file"],
        delimiter=config.get("delimiter", "\t"),
        sequence_separator=config.get("sequence_separator", "|"),
    )
    pair_records = load_pair_records(config["pair_split_dir"])
    validate_gold_paths_against_pair_splits(records, pair_records)
    templates = read_templates(templates_path)
    labels = build_gold_template_labels(records, templates)
    write_split_labels(output_dir, labels)
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/templates.yaml")
    parser.add_argument("--create-toy-data-if-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_gold_template_label_generation(config, create_toy_data_if_missing=args.create_toy_data_if_missing)


if __name__ == "__main__":
    main()
