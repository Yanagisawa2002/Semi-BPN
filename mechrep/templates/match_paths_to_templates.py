"""Match candidate evidence paths to typed mechanism templates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from mechrep.templates.extract_templates import TemplateRecord, read_templates


EVIDENCE_PATH_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "path_id",
    "path_node_ids",
    "path_node_types",
    "path_relation_types",
    "path_score",
)

MATCH_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "path_id",
    "best_template_id",
    "best_template_match_score",
    "node_type_similarity",
    "relation_type_similarity",
    "is_exact_match",
    "path_score",
    "normalized_path_score",
)


@dataclass(frozen=True)
class CandidateEvidencePath:
    pair_id: str
    drug_id: str
    endpoint_id: str
    split: str
    path_id: str
    node_ids: tuple[str, ...]
    node_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    path_score: float | None = None

    @property
    def node_type_sequence(self) -> str:
        return "|".join(self.node_types)

    @property
    def relation_type_sequence(self) -> str:
        return "|".join(self.relation_types)


@dataclass(frozen=True)
class PathTemplateMatch:
    pair_id: str
    drug_id: str
    endpoint_id: str
    split: str
    path_id: str
    best_template_id: str
    best_template_match_score: float
    node_type_similarity: float
    relation_type_similarity: float
    is_exact_match: bool
    path_score: float | None
    normalized_path_score: float | None = None
    template_support_count: int = 0

    def as_row(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "drug_id": self.drug_id,
            "endpoint_id": self.endpoint_id,
            "split": self.split,
            "path_id": self.path_id,
            "best_template_id": self.best_template_id,
            "best_template_match_score": f"{self.best_template_match_score:.8f}",
            "node_type_similarity": f"{self.node_type_similarity:.8f}",
            "relation_type_similarity": f"{self.relation_type_similarity:.8f}",
            "is_exact_match": "1" if self.is_exact_match else "0",
            "path_score": "" if self.path_score is None else f"{self.path_score:.8f}",
            "normalized_path_score": ""
            if self.normalized_path_score is None
            else f"{self.normalized_path_score:.8f}",
        }


def split_sequence(value: str, *, source: str, column: str) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        raise ValueError(f"{source} has empty {column}")
    parts = tuple(part.strip() for part in value.split("|"))
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"{source} has malformed {column}: empty sequence element")
    return parts


def parse_path_score(value: str | None, *, source: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{source} has non-numeric path_score {value!r}") from exc


def candidate_path_from_row(row: dict, *, source: str) -> CandidateEvidencePath:
    required_without_score = [column for column in EVIDENCE_PATH_COLUMNS if column != "path_score"]
    missing = [column for column in required_without_score if row.get(column) in (None, "")]
    if missing:
        raise ValueError(f"{source} has empty required columns: {missing}")
    node_ids = split_sequence(row["path_node_ids"], source=source, column="path_node_ids")
    node_types = split_sequence(row["path_node_types"], source=source, column="path_node_types")
    relation_types = split_sequence(row["path_relation_types"], source=source, column="path_relation_types")
    if len(node_ids) < 2:
        raise ValueError(f"{source} has too-short path_node_ids; at least two nodes are required")
    if len(node_types) != len(node_ids):
        raise ValueError(f"{source} has node_type length {len(node_types)} but node_id length {len(node_ids)}")
    if len(relation_types) != len(node_types) - 1:
        raise ValueError(
            f"{source} has relation_type length {len(relation_types)} but expected node_type length - 1 "
            f"({len(node_types) - 1})"
        )
    return CandidateEvidencePath(
        pair_id=row["pair_id"],
        drug_id=row["drug_id"],
        endpoint_id=row["endpoint_id"],
        split=row["split"],
        path_id=row["path_id"],
        node_ids=node_ids,
        node_types=node_types,
        relation_types=relation_types,
        path_score=parse_path_score(row.get("path_score"), source=source),
    )


def read_candidate_paths(path: str | Path) -> List[CandidateEvidencePath]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = [column for column in EVIDENCE_PATH_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        records = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has more fields than the header")
            records.append(candidate_path_from_row(row, source=f"{path}:{row_number}"))
    if not records:
        raise ValueError(f"{path} contains no candidate evidence paths")
    return records


def sequence_similarity(path_sequence: Sequence[str], template_sequence: Sequence[str]) -> float:
    if not path_sequence or not template_sequence:
        raise ValueError("Cannot compute similarity for empty sequences")
    max_length = max(len(path_sequence), len(template_sequence))
    matches = sum(
        1
        for index in range(min(len(path_sequence), len(template_sequence)))
        if path_sequence[index] == template_sequence[index]
    )
    return matches / max_length


def template_match_score(
    path: CandidateEvidencePath,
    template: TemplateRecord,
    *,
    alpha_node: float = 0.4,
    alpha_relation: float = 0.6,
) -> tuple[float, float, float, bool]:
    node_types = tuple(template.node_type_sequence.split("|"))
    relation_types = tuple(template.relation_type_sequence.split("|"))
    is_exact = path.node_types == node_types and path.relation_types == relation_types
    if is_exact:
        return 1.0, 1.0, 1.0, True
    node_similarity = sequence_similarity(path.node_types, node_types)
    relation_similarity = sequence_similarity(path.relation_types, relation_types)
    score = alpha_node * node_similarity + alpha_relation * relation_similarity
    return score, node_similarity, relation_similarity, False


def match_path_to_templates(
    path: CandidateEvidencePath,
    templates: Sequence[TemplateRecord],
    *,
    alpha_node: float = 0.4,
    alpha_relation: float = 0.6,
) -> PathTemplateMatch:
    if not templates:
        raise ValueError("Cannot match paths without at least one template")
    candidates = []
    for template in templates:
        score, node_similarity, relation_similarity, is_exact = template_match_score(
            path,
            template,
            alpha_node=alpha_node,
            alpha_relation=alpha_relation,
        )
        candidates.append((score, template.support_count, template.template_id, node_similarity, relation_similarity, is_exact))
    score, support_count, template_id, node_similarity, relation_similarity, is_exact = sorted(
        candidates,
        key=lambda item: (-item[0], -item[1], item[2]),
    )[0]
    return PathTemplateMatch(
        pair_id=path.pair_id,
        drug_id=path.drug_id,
        endpoint_id=path.endpoint_id,
        split=path.split,
        path_id=path.path_id,
        best_template_id=template_id,
        best_template_match_score=score,
        node_type_similarity=node_similarity,
        relation_type_similarity=relation_similarity,
        is_exact_match=is_exact,
        path_score=path.path_score,
        template_support_count=support_count,
    )


def match_paths_to_templates(
    paths: Sequence[CandidateEvidencePath],
    templates: Sequence[TemplateRecord],
    *,
    alpha_node: float = 0.4,
    alpha_relation: float = 0.6,
) -> List[PathTemplateMatch]:
    matches = [
        match_path_to_templates(
            path,
            templates,
            alpha_node=alpha_node,
            alpha_relation=alpha_relation,
        )
        for path in paths
    ]
    return sorted(matches, key=lambda item: (item.split, item.pair_id, item.path_id, item.best_template_id))


def write_path_template_matches(path: str | Path, matches: Sequence[PathTemplateMatch]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for match in matches:
            writer.writerow(match.as_row())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-paths", required=True)
    parser.add_argument("--template-table", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--alpha-node", type=float, default=0.4)
    parser.add_argument("--alpha-relation", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = match_paths_to_templates(
        read_candidate_paths(args.candidate_paths),
        read_templates(args.template_table),
        alpha_node=args.alpha_node,
        alpha_relation=args.alpha_relation,
    )
    write_path_template_matches(args.output_tsv, matches)
    print(json.dumps({"num_matches": len(matches)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
