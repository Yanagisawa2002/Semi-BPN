"""Assign high-confidence pseudo-template labels from evidence path matches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

from mechrep.data.build_pairs import SPLITS, PairRecord, load_pair_splits
from mechrep.data.gold_template_dataset import read_gold_template_label_tsv
from mechrep.templates.extract_templates import TemplateRecord, make_template_id, read_templates
from mechrep.templates.match_paths_to_templates import (
    MATCH_COLUMNS,
    CandidateEvidencePath,
    PathTemplateMatch,
    match_paths_to_templates,
    read_candidate_paths,
    write_path_template_matches,
)
from mechrep.templates.template_vocab import TemplateVocab


PREDICTION_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "label",
    "score",
    "split",
    "has_gold_template",
    "primary_template_id",
    "predicted_template_id",
    "template_confidence",
)

PSEUDO_LABEL_COLUMNS = (
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

UNASSIGNED_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "split",
    "label",
    "pair_score",
    "best_template_id",
    "best_template_match_score",
    "final_confidence",
    "reason_unassigned",
)

ASSIGNMENT_SOURCE = "high_confidence_path_template_match"


@dataclass(frozen=True)
class PredictionRow:
    pair_id: str
    drug_id: str
    endpoint_id: str
    label: int
    score: float
    split: str
    predicted_template_id: str
    template_confidence: float


@dataclass(frozen=True)
class AssignmentCandidate:
    match: PathTemplateMatch
    pair_score: float
    final_confidence: float


@dataclass(frozen=True)
class PseudoTemplateAssignment:
    pair_id: str
    drug_id: str
    endpoint_id: str
    split: str
    pseudo_template_id: str
    pseudo_template_index: int | None
    matched_path_id: str
    pair_score: float
    template_match_score: float
    normalized_path_score: float | None
    final_confidence: float

    def as_row(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "drug_id": self.drug_id,
            "endpoint_id": self.endpoint_id,
            "split": self.split,
            "pseudo_template_id": self.pseudo_template_id,
            "pseudo_template_index": "" if self.pseudo_template_index is None else str(self.pseudo_template_index),
            "matched_path_id": self.matched_path_id,
            "pair_score": f"{self.pair_score:.8f}",
            "template_match_score": f"{self.template_match_score:.8f}",
            "normalized_path_score": ""
            if self.normalized_path_score is None
            else f"{self.normalized_path_score:.8f}",
            "final_confidence": f"{self.final_confidence:.8f}",
            "assignment_source": ASSIGNMENT_SOURCE,
        }


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


def _write_tsv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_raw_tsv(path: str | Path, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def read_predictions(path: str | Path) -> Dict[str, PredictionRow]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = [column for column in PREDICTION_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        predictions = {}
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{row_number} has more fields than the header")
            pair_id = row["pair_id"]
            if pair_id in predictions:
                raise ValueError(f"{path} contains duplicate pair_id {pair_id!r}")
            predictions[pair_id] = PredictionRow(
                pair_id=pair_id,
                drug_id=row["drug_id"],
                endpoint_id=row["endpoint_id"],
                label=int(row["label"]),
                score=float(row["score"]),
                split=row["split"],
                predicted_template_id=row["predicted_template_id"],
                template_confidence=float(row["template_confidence"]),
            )
    return predictions


def load_predictions_by_split(config: dict) -> Dict[str, Dict[str, PredictionRow]]:
    prediction_config = config["predictions"]
    paths = {
        "train": prediction_config["train_predictions"],
        "valid": prediction_config["valid_predictions"],
        "test": prediction_config["test_predictions"],
    }
    return {split: read_predictions(path) for split, path in paths.items()}


def validate_predictions_against_pairs(
    pair_splits: Dict[str, Sequence[PairRecord]],
    predictions_by_split: Dict[str, Dict[str, PredictionRow]],
) -> None:
    for split in SPLITS:
        pair_by_id = {pair.pair_id: pair for pair in pair_splits[split]}
        missing = sorted(set(pair_by_id) - set(predictions_by_split[split]))
        if missing:
            raise ValueError(f"Predictions for split {split!r} are missing pair_ids: {missing[:10]}")
        extra = sorted(set(predictions_by_split[split]) - set(pair_by_id))
        if extra:
            raise ValueError(f"Predictions for split {split!r} include unknown pair_ids: {extra[:10]}")
        for pair_id, prediction in predictions_by_split[split].items():
            pair = pair_by_id[pair_id]
            if (
                prediction.drug_id != pair.drug_id
                or prediction.endpoint_id != pair.endpoint_id
                or prediction.label != pair.label
                or prediction.split != split
            ):
                raise ValueError(f"Prediction metadata for pair_id {pair_id!r} does not match {split} pair split")


def load_gold_pair_ids(config: dict) -> Dict[str, set[str]]:
    template_config = config["templates"]
    paths = {
        "train": template_config["gold_labels_train"],
        "valid": template_config["gold_labels_valid"],
        "test": template_config["gold_labels_test"],
    }
    return {split: set(read_gold_template_label_tsv(path)) for split, path in paths.items()}


def load_template_vocab_if_available(path_value: str | None) -> TemplateVocab | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return TemplateVocab.load(path)


def filter_templates_by_vocab(templates: Sequence[TemplateRecord], vocab: TemplateVocab | None) -> List[TemplateRecord]:
    if vocab is None:
        return list(templates)
    filtered = [template for template in templates if template.template_id in vocab.template_id_to_index]
    if not filtered:
        raise ValueError("No templates remain after filtering by template vocabulary")
    return filtered


def evidence_path_config_for_split(config: dict, split: str) -> str | None:
    return config.get("evidence_paths", {}).get(f"{split}_paths")


def load_candidate_paths_by_split(config: dict, splits: Sequence[str]) -> Dict[str, List[CandidateEvidencePath]]:
    paths_by_split = {}
    for split in splits:
        path_value = evidence_path_config_for_split(config, split)
        if not path_value:
            raise ValueError(f"Missing evidence_paths.{split}_paths in config")
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing candidate evidence path file: {path}. "
                "TODO: connect BioPathNet evidence path export to this TSV format."
            )
        paths_by_split[split] = read_candidate_paths(path)
    return paths_by_split


def normalize_path_scores(matches: Sequence[PathTemplateMatch]) -> List[PathTemplateMatch]:
    by_pair = defaultdict(list)
    for match in matches:
        by_pair[match.pair_id].append(match)

    normalized = []
    for pair_id in sorted(by_pair):
        group = sorted(by_pair[pair_id], key=lambda item: item.path_id)
        scores = [match.path_score for match in group if match.path_score is not None]
        min_score = min(scores) if scores else None
        max_score = max(scores) if scores else None
        for match in group:
            if match.path_score is None:
                normalized_score = None
            elif min_score == max_score:
                normalized_score = 1.0
            else:
                normalized_score = (match.path_score - min_score) / (max_score - min_score)
            normalized.append(
                PathTemplateMatch(
                    pair_id=match.pair_id,
                    drug_id=match.drug_id,
                    endpoint_id=match.endpoint_id,
                    split=match.split,
                    path_id=match.path_id,
                    best_template_id=match.best_template_id,
                    best_template_match_score=match.best_template_match_score,
                    node_type_similarity=match.node_type_similarity,
                    relation_type_similarity=match.relation_type_similarity,
                    is_exact_match=match.is_exact_match,
                    path_score=match.path_score,
                    normalized_path_score=normalized_score,
                    template_support_count=match.template_support_count,
                )
            )
    return sorted(normalized, key=lambda item: (item.split, item.pair_id, item.path_id, item.best_template_id))


def compute_final_confidence(
    *,
    pair_score: float,
    template_match_score: float,
    normalized_path_score: float | None,
    w_pair: float,
    w_match: float,
    w_path: float,
) -> float:
    if normalized_path_score is None:
        denominator = w_pair + w_match
        if denominator <= 0:
            raise ValueError("w_pair + w_match must be positive when path_score is missing")
        return (w_pair * pair_score + w_match * template_match_score) / denominator
    return w_pair * pair_score + w_match * template_match_score + w_path * normalized_path_score


def best_assignment_candidates_by_pair(
    matches: Sequence[PathTemplateMatch],
    predictions: Dict[str, PredictionRow],
    *,
    w_pair: float,
    w_match: float,
    w_path: float,
) -> Dict[str, AssignmentCandidate]:
    candidates_by_pair = defaultdict(list)
    for match in matches:
        if match.pair_id not in predictions:
            raise ValueError(f"Evidence path references pair_id absent from predictions: {match.pair_id!r}")
        prediction = predictions[match.pair_id]
        if (
            match.drug_id != prediction.drug_id
            or match.endpoint_id != prediction.endpoint_id
            or match.split != prediction.split
        ):
            raise ValueError(f"Evidence path metadata does not match prediction row for pair_id {match.pair_id!r}")
        pair_score = prediction.score
        final_confidence = compute_final_confidence(
            pair_score=pair_score,
            template_match_score=match.best_template_match_score,
            normalized_path_score=match.normalized_path_score,
            w_pair=w_pair,
            w_match=w_match,
            w_path=w_path,
        )
        candidates_by_pair[match.pair_id].append(
            AssignmentCandidate(match=match, pair_score=pair_score, final_confidence=final_confidence)
        )
    best = {}
    for pair_id, candidates in candidates_by_pair.items():
        best[pair_id] = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.final_confidence,
                -candidate.match.best_template_match_score,
                -(candidate.match.normalized_path_score if candidate.match.normalized_path_score is not None else -1.0),
                -candidate.match.template_support_count,
                candidate.match.best_template_id,
                candidate.match.path_id,
            ),
        )[0]
    return best


def assignment_reason(
    *,
    pair: PairRecord,
    prediction: PredictionRow,
    gold_pair_ids: set[str],
    candidate: AssignmentCandidate | None,
    assignment_splits: set[str],
    tau_pair: float,
    tau_match: float,
    tau_confidence: float,
) -> str | None:
    if pair.split not in assignment_splits:
        return "split_not_allowed"
    if pair.label != 1:
        return "not_positive"
    if pair.pair_id in gold_pair_ids:
        return "already_has_gold_template"
    if prediction.score < tau_pair:
        return "below_tau_pair"
    if candidate is None:
        return "no_candidate_paths"
    if candidate.match.best_template_match_score < tau_match:
        return "below_tau_match"
    if candidate.final_confidence < tau_confidence:
        return "below_tau_confidence"
    return None


def assign_pseudo_templates_for_split(
    *,
    split: str,
    pairs: Sequence[PairRecord],
    predictions: Dict[str, PredictionRow],
    gold_pair_ids: set[str],
    candidates_by_pair: Dict[str, AssignmentCandidate],
    template_vocab: TemplateVocab | None,
    assignment_splits: set[str],
    tau_pair: float,
    tau_match: float,
    tau_confidence: float,
) -> tuple[List[PseudoTemplateAssignment], List[dict]]:
    assignments = []
    unassigned = []
    for pair in sorted(pairs, key=lambda item: item.pair_id):
        prediction = predictions[pair.pair_id]
        candidate = candidates_by_pair.get(pair.pair_id)
        reason = assignment_reason(
            pair=pair,
            prediction=prediction,
            gold_pair_ids=gold_pair_ids,
            candidate=candidate,
            assignment_splits=assignment_splits,
            tau_pair=tau_pair,
            tau_match=tau_match,
            tau_confidence=tau_confidence,
        )
        if reason is None and candidate is not None:
            template_id = candidate.match.best_template_id
            assignments.append(
                PseudoTemplateAssignment(
                    pair_id=pair.pair_id,
                    drug_id=pair.drug_id,
                    endpoint_id=pair.endpoint_id,
                    split=split,
                    pseudo_template_id=template_id,
                    pseudo_template_index=template_vocab.index(template_id) if template_vocab is not None else None,
                    matched_path_id=candidate.match.path_id,
                    pair_score=prediction.score,
                    template_match_score=candidate.match.best_template_match_score,
                    normalized_path_score=candidate.match.normalized_path_score,
                    final_confidence=candidate.final_confidence,
                )
            )
            continue
        unassigned.append(
            {
                "pair_id": pair.pair_id,
                "drug_id": pair.drug_id,
                "endpoint_id": pair.endpoint_id,
                "split": split,
                "label": str(pair.label),
                "pair_score": f"{prediction.score:.8f}",
                "best_template_id": "" if candidate is None else candidate.match.best_template_id,
                "best_template_match_score": ""
                if candidate is None
                else f"{candidate.match.best_template_match_score:.8f}",
                "final_confidence": "" if candidate is None else f"{candidate.final_confidence:.8f}",
                "reason_unassigned": reason or "unknown",
            }
        )
    return assignments, unassigned


def make_assignment_report(
    *,
    train_pairs: Sequence[PairRecord],
    gold_pair_ids_by_split: Dict[str, set[str]],
    assignments: Sequence[PseudoTemplateAssignment],
    assignment_splits: Sequence[str],
    threshold_config: dict,
    match_by_path_id: Dict[str, PathTemplateMatch],
) -> dict:
    train_gold_ids = gold_pair_ids_by_split["train"]
    train_positive = [pair for pair in train_pairs if pair.label == 1]
    train_no_gold_positive = [pair for pair in train_positive if pair.pair_id not in train_gold_ids]
    assigned_train = [assignment for assignment in assignments if assignment.split == "train"]
    template_distribution = Counter(assignment.pseudo_template_id for assignment in assignments)

    def average(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    leakage_checks = {
        "number_of_assigned_test_pairs": sum(1 for assignment in assignments if assignment.split == "test"),
        "number_of_assigned_valid_pairs": sum(1 for assignment in assignments if assignment.split == "valid"),
        "number_of_assigned_negative_pairs": 0,
        "number_of_assigned_gold_template_pairs": 0,
    }
    train_pair_by_id = {pair.pair_id: pair for pair in train_pairs}
    for assignment in assignments:
        if assignment.split == "train":
            pair = train_pair_by_id[assignment.pair_id]
            if pair.label != 1:
                leakage_checks["number_of_assigned_negative_pairs"] += 1
            if assignment.pair_id in train_gold_ids:
                leakage_checks["number_of_assigned_gold_template_pairs"] += 1

    exact_values = []
    for assignment in assignments:
        match = match_by_path_id.get(f"{assignment.pair_id}\t{assignment.matched_path_id}")
        if match is not None:
            exact_values.append(1.0 if match.is_exact_match else 0.0)

    report = {
        "total_train_pairs": len(train_pairs),
        "total_train_positive_pairs": len(train_positive),
        "total_train_gold_template_pairs": len(train_gold_ids),
        "total_train_no_gold_positive_pairs": len(train_no_gold_positive),
        "assigned_no_gold_positive_pairs": len(assigned_train),
        "assignment_coverage": len(assigned_train) / len(train_no_gold_positive) if train_no_gold_positive else 0.0,
        "average_pair_score_assigned": average([assignment.pair_score for assignment in assignments]),
        "average_template_match_score_assigned": average(
            [assignment.template_match_score for assignment in assignments]
        ),
        "average_final_confidence_assigned": average([assignment.final_confidence for assignment in assignments]),
        "template_distribution": dict(sorted(template_distribution.items())),
        "exact_match_fraction": average(exact_values),
        "threshold_config": threshold_config,
        "assignment_splits": list(assignment_splits),
        "leakage_checks": leakage_checks,
    }
    if leakage_checks["number_of_assigned_test_pairs"] != 0:
        raise ValueError("Leakage check failed: assigned test pairs must be zero")
    if "valid" not in assignment_splits and leakage_checks["number_of_assigned_valid_pairs"] != 0:
        raise ValueError("Leakage check failed: assigned valid pairs must be zero unless valid is enabled")
    if leakage_checks["number_of_assigned_negative_pairs"] != 0:
        raise ValueError("Leakage check failed: assigned negative pairs must be zero")
    if leakage_checks["number_of_assigned_gold_template_pairs"] != 0:
        raise ValueError("Leakage check failed: assigned gold-template pairs must be zero")
    return report


def create_toy_inputs(config: dict) -> None:
    output_dir = Path(config["experiment"]["output_dir"])
    toy_dir = output_dir / "toy_inputs"
    split_dir = toy_dir / "splits"
    template_dir = toy_dir / "templates"
    prediction_dir = toy_dir / "predictions"
    evidence_dir = toy_dir / "evidence_paths"

    config["data"]["split_dir"] = str(split_dir)
    config["data"]["train_file"] = str(split_dir / "train.tsv")
    config["data"]["valid_file"] = str(split_dir / "valid.tsv")
    config["data"]["test_file"] = str(split_dir / "test.tsv")
    config["templates"]["template_table"] = str(template_dir / "templates.tsv")
    config["templates"]["template_vocab"] = str(template_dir / "template_vocab.json")
    config["templates"]["gold_labels_train"] = str(template_dir / "gold_template_labels_train.tsv")
    config["templates"]["gold_labels_valid"] = str(template_dir / "gold_template_labels_valid.tsv")
    config["templates"]["gold_labels_test"] = str(template_dir / "gold_template_labels_test.tsv")
    config["predictions"]["train_predictions"] = str(prediction_dir / "predictions_train.tsv")
    config["predictions"]["valid_predictions"] = str(prediction_dir / "predictions_valid.tsv")
    config["predictions"]["test_predictions"] = str(prediction_dir / "predictions_test.tsv")
    config["evidence_paths"]["train_paths"] = str(evidence_dir / "evidence_paths_train.tsv")

    pair_columns = ["pair_id", "drug_id", "endpoint_id", "label"]
    _write_raw_tsv(
        split_dir / "train.tsv",
        pair_columns,
        [
            ("P_gold", "D1", "E_train", 1),
            ("P_pseudo", "D2", "E_train", 1),
            ("P_low_pair", "D3", "E_train", 1),
            ("P_no_path", "D4", "E_train", 1),
            ("P_negative", "D5", "E_train", 0),
        ],
    )
    _write_raw_tsv(split_dir / "valid.tsv", pair_columns, [("P_valid", "D2", "E_valid", 1)])
    _write_raw_tsv(split_dir / "test.tsv", pair_columns, [("P_test", "D2", "E_test", 1)])

    template_exact = make_template_id("Drug|Gene|Endpoint", "targets|associated_with")
    template_other = make_template_id("Drug|Pathway|Endpoint", "participates_in|associated_with")
    _write_raw_tsv(
        template_dir / "templates.tsv",
        ["template_id", "node_type_sequence", "relation_type_sequence", "support_count", "pair_ids", "path_ids"],
        [
            (template_exact, "Drug|Gene|Endpoint", "targets|associated_with", 4, "P_gold", "path_gold"),
            (template_other, "Drug|Pathway|Endpoint", "participates_in|associated_with", 1, "P_gold", "path_other"),
        ],
    )
    TemplateVocab.from_template_ids([template_exact, template_other]).save(template_dir / "template_vocab.json")

    label_columns = [
        "pair_id",
        "drug_id",
        "endpoint_id",
        "split",
        "template_ids",
        "primary_template_id",
        "num_gold_paths",
    ]
    _write_raw_tsv(
        template_dir / "gold_template_labels_train.tsv",
        label_columns,
        [("P_gold", "D1", "E_train", "train", template_exact, template_exact, 1)],
    )
    _write_raw_tsv(template_dir / "gold_template_labels_valid.tsv", label_columns, [])
    _write_raw_tsv(template_dir / "gold_template_labels_test.tsv", label_columns, [])

    prediction_columns = list(PREDICTION_COLUMNS) + ["template_top3_ids"]
    _write_raw_tsv(
        prediction_dir / "predictions_train.tsv",
        prediction_columns,
        [
            ("P_gold", "D1", "E_train", 1, 0.98, "train", 1, template_exact, template_exact, 0.99, template_exact),
            ("P_pseudo", "D2", "E_train", 1, 0.95, "train", 0, "", template_exact, 0.98, template_exact),
            ("P_low_pair", "D3", "E_train", 1, 0.20, "train", 0, "", template_exact, 0.98, template_exact),
            ("P_no_path", "D4", "E_train", 1, 0.95, "train", 0, "", template_exact, 0.98, template_exact),
            ("P_negative", "D5", "E_train", 0, 0.97, "train", 0, "", template_exact, 0.98, template_exact),
        ],
    )
    _write_raw_tsv(
        prediction_dir / "predictions_valid.tsv",
        prediction_columns,
        [("P_valid", "D2", "E_valid", 1, 0.95, "valid", 0, "", template_exact, 0.98, template_exact)],
    )
    _write_raw_tsv(
        prediction_dir / "predictions_test.tsv",
        prediction_columns,
        [("P_test", "D2", "E_test", 1, 0.95, "test", 0, "", template_exact, 0.98, template_exact)],
    )

    _write_raw_tsv(
        evidence_dir / "evidence_paths_train.tsv",
        [
            "pair_id",
            "drug_id",
            "endpoint_id",
            "split",
            "path_id",
            "path_node_ids",
            "path_node_types",
            "path_relation_types",
            "path_score",
        ],
        [
            ("P_pseudo", "D2", "E_train", "train", "path_pseudo", "D2|G1|E_train", "Drug|Gene|Endpoint", "targets|associated_with", 0.9),
            ("P_low_pair", "D3", "E_train", "train", "path_low", "D3|G1|E_train", "Drug|Gene|Endpoint", "targets|associated_with", 0.9),
            ("P_negative", "D5", "E_train", "train", "path_negative", "D5|G1|E_train", "Drug|Gene|Endpoint", "targets|associated_with", 0.9),
        ],
    )


def ensure_inputs(config: dict, *, create_toy_data_if_missing: bool) -> None:
    required = [
        Path(config["data"]["split_dir"]) / "train.tsv",
        Path(config["data"]["split_dir"]) / "valid.tsv",
        Path(config["data"]["split_dir"]) / "test.tsv",
        Path(config["templates"]["template_table"]),
        Path(config["templates"]["gold_labels_train"]),
        Path(config["templates"]["gold_labels_valid"]),
        Path(config["templates"]["gold_labels_test"]),
        Path(config["predictions"]["train_predictions"]),
        Path(config["predictions"]["valid_predictions"]),
        Path(config["predictions"]["test_predictions"]),
        Path(config["evidence_paths"]["train_paths"]),
    ]
    if all(path.exists() for path in required):
        return
    if create_toy_data_if_missing:
        create_toy_inputs(config)
        return
    missing = [str(path) for path in required if not path.exists()]
    raise FileNotFoundError(
        "Missing pseudo-template assignment inputs: "
        f"{missing}. TODO: export candidate evidence paths to evidence_paths_train.tsv."
    )


def run_pseudo_assignment(config: dict, *, create_toy_data_if_missing: bool = False) -> dict:
    ensure_inputs(config, create_toy_data_if_missing=create_toy_data_if_missing)
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    assignment_config = config["pseudo_assignment"]
    assignment_splits = list(assignment_config.get("assignment_splits", ["train"]))
    unknown_splits = sorted(set(assignment_splits) - set(SPLITS))
    if unknown_splits:
        raise ValueError(f"Unknown assignment_splits values: {unknown_splits}")
    if "test" in assignment_splits:
        raise ValueError("assignment_splits must not include test; test paths and labels cannot be used for pseudo assignment")

    pair_splits = load_pair_splits(config["data"]["split_dir"])
    predictions_by_split = load_predictions_by_split(config)
    validate_predictions_against_pairs(pair_splits, predictions_by_split)
    gold_pair_ids_by_split = load_gold_pair_ids(config)

    template_vocab = load_template_vocab_if_available(config["templates"].get("template_vocab"))
    templates = filter_templates_by_vocab(read_templates(config["templates"]["template_table"]), template_vocab)
    template_support = {template.template_id: template.support_count for template in templates}

    paths_by_split = load_candidate_paths_by_split(config, assignment_splits)
    all_matches = []
    for split in assignment_splits:
        matches = match_paths_to_templates(
            paths_by_split[split],
            templates,
            alpha_node=float(config.get("matching", {}).get("alpha_node", 0.4)),
            alpha_relation=float(config.get("matching", {}).get("alpha_relation", 0.6)),
        )
        normalized_matches = normalize_path_scores(matches)
        all_matches.extend(normalized_matches)
        write_path_template_matches(output_dir / f"path_template_matches_{split}.tsv", normalized_matches)

    candidates_by_split = {}
    for split in assignment_splits:
        split_matches = [match for match in all_matches if match.split == split]
        candidates_by_split[split] = best_assignment_candidates_by_pair(
            split_matches,
            predictions_by_split[split],
            w_pair=float(assignment_config.get("w_pair", 0.4)),
            w_match=float(assignment_config.get("w_match", 0.4)),
            w_path=float(assignment_config.get("w_path", 0.2)),
        )

    all_assignments = []
    unassigned_by_split = {}
    for split in SPLITS:
        candidates = candidates_by_split.get(split, {})
        assignments, unassigned = assign_pseudo_templates_for_split(
            split=split,
            pairs=pair_splits[split],
            predictions=predictions_by_split[split],
            gold_pair_ids=gold_pair_ids_by_split[split],
            candidates_by_pair=candidates,
            template_vocab=template_vocab,
            assignment_splits=set(assignment_splits),
            tau_pair=float(assignment_config.get("tau_pair", 0.7)),
            tau_match=float(assignment_config.get("tau_match", 0.8)),
            tau_confidence=float(assignment_config.get("tau_confidence", 0.75)),
        )
        all_assignments.extend(assignments)
        unassigned_by_split[split] = unassigned

    train_assignments = [assignment for assignment in all_assignments if assignment.split == "train"]
    _write_tsv(
        output_dir / "pseudo_template_labels_train.tsv",
        PSEUDO_LABEL_COLUMNS,
        [assignment.as_row() for assignment in train_assignments],
    )
    _write_tsv(
        output_dir / "unassigned_no_gold_positives_train.tsv",
        UNASSIGNED_COLUMNS,
        unassigned_by_split["train"],
    )

    match_by_path_id = {f"{match.pair_id}\t{match.path_id}": match for match in all_matches}
    threshold_config = {
        "tau_pair": float(assignment_config.get("tau_pair", 0.7)),
        "tau_match": float(assignment_config.get("tau_match", 0.8)),
        "tau_confidence": float(assignment_config.get("tau_confidence", 0.75)),
        "w_pair": float(assignment_config.get("w_pair", 0.4)),
        "w_match": float(assignment_config.get("w_match", 0.4)),
        "w_path": float(assignment_config.get("w_path", 0.2)),
    }
    report = make_assignment_report(
        train_pairs=pair_splits["train"],
        gold_pair_ids_by_split=gold_pair_ids_by_split,
        assignments=all_assignments,
        assignment_splits=assignment_splits,
        threshold_config=threshold_config,
        match_by_path_id=match_by_path_id,
    )
    report["template_support_used"] = template_support
    with (output_dir / "assignment_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pseudo_template.yaml")
    parser.add_argument("--create-toy-data-if-missing", action="store_true")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("experiment", {})["output_dir"] = args.output_dir
    report = run_pseudo_assignment(config, create_toy_data_if_missing=args.create_toy_data_if_missing)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
