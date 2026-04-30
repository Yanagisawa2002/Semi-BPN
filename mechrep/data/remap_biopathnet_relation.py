"""Create a BioPathNet debug dataset with remapped supervision relation names."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Sequence

import yaml


BIOPATHNET_FILES = ("train1.txt", "train2.txt", "valid.txt", "test.txt")
DEFAULT_TARGET_FILES = ("train2.txt", "valid.txt", "test.txt")
DEFAULT_COPY_FILES = ("entity_types.txt", "entity_names.txt", "prepare_report.json", "path_subgraph_report.json")


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


def _read_triplets(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(f"{path}:{row_number} must have exactly three tab-separated columns")
            rows.append((row[0], row[1], row[2]))
    return rows


def _write_triplets(path: Path, rows: Sequence[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def _relation_counts(rows: Sequence[tuple[str, str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, relation, _ in rows:
        counts[relation] = counts.get(relation, 0) + 1
    return dict(sorted(counts.items()))


def _remap_rows(
    rows: Sequence[tuple[str, str, str]],
    *,
    source_relation: str,
    target_relation: str,
    require_source_relation: bool,
    source: Path,
) -> tuple[list[tuple[str, str, str]], dict]:
    remapped = []
    unexpected_relations: dict[str, int] = {}
    remapped_count = 0
    for head, relation, tail in rows:
        if relation == source_relation:
            remapped.append((head, target_relation, tail))
            remapped_count += 1
            continue
        unexpected_relations[relation] = unexpected_relations.get(relation, 0) + 1
        remapped.append((head, relation, tail))

    if require_source_relation and unexpected_relations:
        raise ValueError(
            f"{source} contains relations other than {source_relation!r}: "
            f"{dict(sorted(unexpected_relations.items()))}"
        )

    return remapped, {
        "rows": len(rows),
        "source_relation_rows": remapped_count,
        "unchanged_rows": len(rows) - remapped_count,
        "unexpected_relation_counts": dict(sorted(unexpected_relations.items())),
    }


def remap_biopathnet_relation(config: dict) -> dict:
    """Copy a BioPathNet dataset while remapping target split relation names.

    This is intended for protocol-parity diagnostics. It does not alter graph
    topology or split membership; only the supervision relation label changes
    in configured target files.
    """

    source_dir = Path(config["source_dir"])
    output_dir = Path(config["output_dir"])
    source_relation = str(config.get("source_relation", "affects_endpoint"))
    target_relation = str(config.get("target_relation", "indication"))
    files = tuple(config.get("files", BIOPATHNET_FILES))
    target_files = set(config.get("target_files", DEFAULT_TARGET_FILES))
    copy_files = tuple(config.get("copy_files", DEFAULT_COPY_FILES))
    require_source_relation = bool(config.get("require_source_relation", True))
    forbid_source_relation_in_copied_files = bool(config.get("forbid_source_relation_in_copied_files", True))
    forbid_target_relation_in_copied_files = bool(config.get("forbid_target_relation_in_copied_files", True))

    if not source_relation:
        raise ValueError("source_relation must be non-empty")
    if not target_relation:
        raise ValueError("target_relation must be non-empty")
    if source_relation == target_relation:
        raise ValueError("source_relation and target_relation must differ")
    for file_name in files:
        source_path = source_dir / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing BioPathNet input file: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_reports = {}
    copied_file_reports = {}

    for file_name in files:
        source_path = source_dir / file_name
        output_path = output_dir / file_name
        if file_name in target_files:
            rows = _read_triplets(source_path)
            output_rows, stats = _remap_rows(
                rows,
                source_relation=source_relation,
                target_relation=target_relation,
                require_source_relation=require_source_relation,
                source=source_path,
            )
            _write_triplets(output_path, output_rows)
            stats["input_relation_counts"] = _relation_counts(rows)
            stats["output_relation_counts"] = _relation_counts(output_rows)
            file_reports[file_name] = stats
        else:
            shutil.copy2(source_path, output_path)
            rows = _read_triplets(source_path)
            file_reports[file_name] = {
                "rows": len(rows),
                "copied_without_remap": True,
                "relation_counts": _relation_counts(rows),
                "source_relation_rows": sum(1 for _, relation, _ in rows if relation == source_relation),
                "target_relation_rows": sum(1 for _, relation, _ in rows if relation == target_relation),
            }

    for file_name in copy_files:
        source_path = source_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / file_name)
            copied_file_reports[file_name] = "copied"
        else:
            copied_file_reports[file_name] = "missing"

    target_file_rows = sum(report["rows"] for name, report in file_reports.items() if name in target_files)
    target_file_remapped_rows = sum(
        report.get("source_relation_rows", 0) for name, report in file_reports.items() if name in target_files
    )
    copied_target_relation_rows = sum(
        report.get("target_relation_rows", 0) for name, report in file_reports.items() if name not in target_files
    )
    copied_source_relation_rows = sum(
        report.get("source_relation_rows", 0) for name, report in file_reports.items() if name not in target_files
    )
    if forbid_source_relation_in_copied_files and copied_source_relation_rows:
        raise ValueError(
            f"Copied files contain {copied_source_relation_rows} {source_relation!r} rows; "
            "target supervision edges must not appear in the BioPathNet fact graph"
        )
    if forbid_target_relation_in_copied_files and copied_target_relation_rows:
        raise ValueError(
            f"Copied files contain {copied_target_relation_rows} {target_relation!r} rows; "
            "target supervision edges must not appear in the BioPathNet fact graph"
        )

    report = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_relation": source_relation,
        "target_relation": target_relation,
        "files": list(files),
        "target_files": sorted(target_files),
        "require_source_relation": require_source_relation,
        "forbid_source_relation_in_copied_files": forbid_source_relation_in_copied_files,
        "forbid_target_relation_in_copied_files": forbid_target_relation_in_copied_files,
        "target_file_rows": target_file_rows,
        "target_file_remapped_rows": target_file_remapped_rows,
        "copied_files": copied_file_reports,
        "file_reports": file_reports,
        "leakage_checks": {
            "source_relation_rows_in_copied_files": copied_source_relation_rows,
            "target_relation_rows_in_copied_files": copied_target_relation_rows,
        },
    }
    with (output_dir / "relation_remap_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/biopathnet_indication_debug_prepare.yaml")
    args = parser.parse_args()
    report = remap_biopathnet_relation(load_config(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
