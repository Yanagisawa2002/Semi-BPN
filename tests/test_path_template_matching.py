import csv
import shutil
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.templates.extract_templates import TemplateRecord
from mechrep.templates.match_paths_to_templates import (
    candidate_path_from_row,
    match_path_to_templates,
    read_candidate_paths,
)


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"path_template_matching_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _template(template_id, node_types, relations, support=1):
    return TemplateRecord(
        template_id=template_id,
        node_type_sequence=node_types,
        relation_type_sequence=relations,
        support_count=support,
        pair_ids=("P1",),
        path_ids=("path1",),
    )


def _path(node_types="Drug|Gene|Endpoint", relations="targets|associated_with", path_id="path1"):
    node_count = len(node_types.split("|"))
    node_ids = "|".join(["D1"] + [f"N{index}" for index in range(1, node_count - 1)] + ["E1"])
    return candidate_path_from_row(
        {
            "pair_id": "P1",
            "drug_id": "D1",
            "endpoint_id": "E1",
            "split": "train",
            "path_id": path_id,
            "path_node_ids": node_ids,
            "path_node_types": node_types,
            "path_relation_types": relations,
            "path_score": "0.8",
        },
        source="toy",
    )


def test_exact_match_returns_score_one():
    match = match_path_to_templates(
        _path(),
        [_template("T_exact", "Drug|Gene|Endpoint", "targets|associated_with")],
    )

    assert match.best_template_id == "T_exact"
    assert match.best_template_match_score == 1.0
    assert match.node_type_similarity == 1.0
    assert match.relation_type_similarity == 1.0
    assert match.is_exact_match is True


def test_partial_match_computes_deterministic_similarity_with_length_mismatch():
    match = match_path_to_templates(
        _path(node_types="Drug|Gene|Pathway|Endpoint", relations="targets|participates_in|associated_with"),
        [_template("T_short", "Drug|Gene|Endpoint", "targets|associated_with")],
        alpha_node=0.4,
        alpha_relation=0.6,
    )

    assert match.node_type_similarity == 2 / 4
    assert match.relation_type_similarity == 1 / 3
    assert match.best_template_match_score == pytest.approx(0.4 * (2 / 4) + 0.6 * (1 / 3))
    assert match.is_exact_match is False


def test_relation_and_node_mismatch_lower_score():
    relation_mismatch = match_path_to_templates(
        _path(relations="binds|associated_with"),
        [_template("T", "Drug|Gene|Endpoint", "targets|associated_with")],
    )
    node_mismatch = match_path_to_templates(
        _path(node_types="Drug|Pathway|Endpoint"),
        [_template("T", "Drug|Gene|Endpoint", "targets|associated_with")],
    )

    assert relation_mismatch.relation_type_similarity == 0.5
    assert relation_mismatch.best_template_match_score < 1.0
    assert node_mismatch.node_type_similarity == pytest.approx(2 / 3)
    assert node_mismatch.best_template_match_score < 1.0


def test_tie_breaking_prefers_support_then_lexical_template_id():
    path = _path()
    support_match = match_path_to_templates(
        path,
        [
            _template("T_b", "Drug|Pathway|Endpoint", "targets|associated_with", support=1),
            _template("T_a", "Drug|Pathway|Endpoint", "targets|associated_with", support=3),
        ],
    )
    lexical_match = match_path_to_templates(
        path,
        [
            _template("T_b", "Drug|Pathway|Endpoint", "targets|associated_with", support=1),
            _template("T_a", "Drug|Pathway|Endpoint", "targets|associated_with", support=1),
        ],
    )

    assert support_match.best_template_id == "T_a"
    assert lexical_match.best_template_id == "T_a"


def test_malformed_paths_raise_clear_errors():
    with pytest.raises(ValueError, match="node_type length"):
        candidate_path_from_row(
            {
                "pair_id": "P1",
                "drug_id": "D1",
                "endpoint_id": "E1",
                "split": "train",
                "path_id": "bad",
                "path_node_ids": "D1|G1|E1",
                "path_node_types": "Drug|Gene",
                "path_relation_types": "targets|associated_with",
                "path_score": "0.8",
            },
            source="bad",
        )

    with pytest.raises(ValueError, match="relation_type length"):
        candidate_path_from_row(
            {
                "pair_id": "P1",
                "drug_id": "D1",
                "endpoint_id": "E1",
                "split": "train",
                "path_id": "bad",
                "path_node_ids": "D1|G1|E1",
                "path_node_types": "Drug|Gene|Endpoint",
                "path_relation_types": "targets",
                "path_score": "0.8",
            },
            source="bad",
        )


def test_read_candidate_paths_rejects_empty_sequence_elements():
    tmp_dir = _workspace_tmp_dir()
    try:
        path = tmp_dir / "paths.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
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
                ]
            )
            writer.writerow(["P1", "D1", "E1", "train", "bad", "D1||E1", "Drug||Endpoint", "targets|x", "0.1"])

        with pytest.raises(ValueError, match="empty sequence element"):
            read_candidate_paths(path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
