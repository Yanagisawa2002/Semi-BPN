"""Prepare BioPathNet full-model input files with complete entity metadata."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Sequence

import yaml


BIOPATHNET_FILES = ("train1.txt", "train2.txt", "valid.txt", "test.txt")
ENTITY_TYPE_FILE = "entity_types.txt"
ENTITY_NAME_FILE = "entity_names.txt"


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


def read_type_vocab(conversion_report_path: str | Path) -> Dict[str, int]:
    with Path(conversion_report_path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    type_vocab = report.get("entity_report", {}).get("type_vocab")
    if not isinstance(type_vocab, dict):
        raise ValueError(f"{conversion_report_path} is missing entity_report.type_vocab")
    return {str(name): int(index) for name, index in type_vocab.items()}


def iter_triple_entities(paths: Sequence[str | Path]) -> Iterable[str]:
    for path in paths:
        path = Path(path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row_number, row in enumerate(reader, start=1):
                if len(row) != 3:
                    raise ValueError(f"{path}:{row_number} must have exactly three tab-separated columns")
                yield row[0]
                yield row[2]


def read_two_column_tsv(path: str | Path) -> Dict[str, str]:
    values = {}
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"{path}:{row_number} must have exactly two tab-separated columns")
            key, value = row
            if key in values:
                raise ValueError(f"{path} contains duplicate entity {key!r}")
            values[key] = value
    return values


def infer_missing_entity_type(entity_id: str, type_vocab: Dict[str, int]) -> int:
    if entity_id.startswith("disease:"):
        return type_vocab["Disease"]
    if entity_id.startswith("drug:"):
        return type_vocab["Drug"]
    if entity_id.startswith("gene/protein:"):
        return type_vocab["Gene"]
    raise ValueError(
        f"Cannot infer BioPathNet entity type for missing entity {entity_id!r}; "
        "extend the preparation config or source metadata before running full BioPathNet"
    )


def write_two_column_tsv(path: str | Path, rows: Sequence[tuple[str, str | int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def prepare_biopathnet_full(config: dict) -> dict:
    source_dir = Path(config["source_dir"])
    output_dir = Path(config["output_dir"])
    conversion_report = Path(config["conversion_report"])
    files = tuple(config.get("files", BIOPATHNET_FILES))
    for file_name in files:
        source_path = source_dir / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing BioPathNet input file: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name in files:
        shutil.copy2(source_dir / file_name, output_dir / file_name)

    type_vocab = read_type_vocab(conversion_report)
    entity_types = read_two_column_tsv(source_dir / ENTITY_TYPE_FILE)
    entity_names = read_two_column_tsv(source_dir / ENTITY_NAME_FILE)
    triple_entities = set(iter_triple_entities(source_dir / file_name for file_name in files))

    missing_type_entities = sorted(triple_entities - set(entity_types))
    missing_name_entities = sorted(triple_entities - set(entity_names))
    repaired_types = dict(entity_types)
    repaired_names = dict(entity_names)

    for entity_id in missing_type_entities:
        repaired_types[entity_id] = str(infer_missing_entity_type(entity_id, type_vocab))
    for entity_id in missing_name_entities:
        repaired_names[entity_id] = entity_id

    type_rows = sorted(repaired_types.items(), key=lambda item: item[0])
    name_rows = sorted(repaired_names.items(), key=lambda item: item[0])
    write_two_column_tsv(output_dir / ENTITY_TYPE_FILE, type_rows)
    write_two_column_tsv(output_dir / ENTITY_NAME_FILE, name_rows)

    report = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "files": list(files),
        "num_triple_entities": len(triple_entities),
        "original_entity_types": len(entity_types),
        "original_entity_names": len(entity_names),
        "missing_type_entities_added": len(missing_type_entities),
        "missing_name_entities_added": len(missing_name_entities),
        "missing_type_entity_examples": missing_type_entities[:20],
        "missing_name_entity_examples": missing_name_entities[:20],
        "final_entity_types": len(repaired_types),
        "final_entity_names": len(repaired_names),
    }
    with (output_dir / "prepare_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/biopathnet_full_prepare.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_biopathnet_full(load_config(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
