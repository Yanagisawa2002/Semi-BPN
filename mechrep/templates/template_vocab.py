"""Deterministic template vocabulary utilities."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

from mechrep.templates.extract_templates import TemplateRecord, read_templates


LABEL_TEMPLATE_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "template_ids",
    "primary_template_id",
    "num_gold_paths",
)


@dataclass(frozen=True)
class TemplateVocab:
    template_id_to_index: Dict[str, int]
    template_index_to_id: Dict[int, str]

    @classmethod
    def from_template_ids(cls, template_ids: Iterable[str]) -> "TemplateVocab":
        ordered = sorted(set(template_ids))
        if not ordered:
            raise ValueError("Template vocabulary cannot be empty")
        id_to_index = {template_id: index for index, template_id in enumerate(ordered)}
        return cls(
            template_id_to_index=id_to_index,
            template_index_to_id={index: template_id for template_id, index in id_to_index.items()},
        )

    @property
    def size(self) -> int:
        return len(self.template_id_to_index)

    def index(self, template_id: str) -> int:
        if template_id not in self.template_id_to_index:
            raise KeyError(f"Unknown template_id {template_id!r}")
        return self.template_id_to_index[template_id]

    def template_id(self, index: int) -> str:
        if index not in self.template_index_to_id:
            raise KeyError(f"Unknown template index {index!r}")
        return self.template_index_to_id[index]

    def as_json_dict(self) -> dict:
        return {
            "template_id_to_index": self.template_id_to_index,
            "template_index_to_id": {
                str(index): template_id for index, template_id in sorted(self.template_index_to_id.items())
            },
            "num_templates": self.size,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.as_json_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "TemplateVocab":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        id_to_index = payload.get("template_id_to_index")
        if not isinstance(id_to_index, dict) or not id_to_index:
            raise ValueError(f"{path} does not contain a non-empty template_id_to_index mapping")
        normalized = {str(template_id): int(index) for template_id, index in id_to_index.items()}
        if sorted(normalized.values()) != list(range(len(normalized))):
            raise ValueError(f"{path} template indices must be contiguous and start at zero")
        return cls(
            template_id_to_index=normalized,
            template_index_to_id={index: template_id for template_id, index in normalized.items()},
        )


def read_label_template_ids(path: str | Path) -> set[str]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = [column for column in LABEL_TEMPLATE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        template_ids = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has more fields than the header")
            for template_id in filter(None, row["template_ids"].split("|")):
                template_ids.add(template_id)
            if row["primary_template_id"]:
                template_ids.add(row["primary_template_id"])
        return template_ids


def build_template_vocab(
    template_table: str | Path,
    *,
    train_labels: str | Path | None = None,
    use_train_templates_only: bool = True,
) -> TemplateVocab:
    templates = read_templates(template_table)
    template_ids_from_table = {template.template_id for template in templates}
    if len(template_ids_from_table) != len(templates):
        raise ValueError(f"{template_table} contains duplicate template_id values")

    if use_train_templates_only:
        if train_labels is None:
            allowed_ids = set(template_ids_from_table)
        else:
            allowed_ids = read_label_template_ids(train_labels)
            unknown = sorted(allowed_ids - template_ids_from_table)
            if unknown:
                raise ValueError(
                    f"Training labels reference template IDs not present in template table: {unknown[:10]}"
                )
        selected_ids = sorted(template_ids_from_table & allowed_ids)
    else:
        selected_ids = sorted(template_ids_from_table)

    if not selected_ids:
        raise ValueError("No templates are available after applying the training template filter")
    return TemplateVocab.from_template_ids(selected_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-table", required=True)
    parser.add_argument("--train-labels")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--include-all-templates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab = build_template_vocab(
        args.template_table,
        train_labels=args.train_labels,
        use_train_templates_only=not args.include_all_templates,
    )
    vocab.save(args.output_json)


if __name__ == "__main__":
    main()
