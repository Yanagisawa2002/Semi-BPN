"""Train original BioPathNet with explicit endpoint pair labels.

This trainer keeps the original BioPathNet / NBFNet encoder intact, but replaces
BioPathNet's internal random corruption objective with direct BCE on our
positive and negative endpoint pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pprint
import random
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from mechrep.data.build_pairs import PairRecord, read_pair_tsv
from mechrep.data.gold_template_dataset import (
    IGNORE_TEMPLATE_INDEX,
    GoldTemplateLabelRow,
    read_gold_template_label_tsv,
)
from mechrep.data.semi_template_dataset import (
    PseudoTemplateLabelRow,
    cap_pseudo_labels_by_template,
    read_pseudo_template_label_tsv,
)
from mechrep.evaluation.eval_prediction import evaluate_predictions, write_metrics_json
from mechrep.evaluation.score_original_biopathnet_pairs import _write_predictions, score_records
from mechrep.templates.template_vocab import TemplateVocab
from mechrep.training.progress import EarlyStopper, iter_progress
from mechrep.training.run_original_biopathnet_linux import (
    _solver_load_without_graphs,
    prepare_environment,
    repository_root,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairwiseTrainingOptions:
    split_dir: Path
    relation_name: str
    train_batch_size: int
    eval_batch_size: int
    validation_interval: int
    selection_metric: str
    group_by: str | None
    k_values: tuple[int, ...]
    progress_bar: bool
    progress_log_interval: int
    early_stop_patience: int | None
    early_stop_min_delta: float
    shuffle: bool
    final_splits: tuple[str, ...]


@dataclass(frozen=True)
class TemplateTrainingOptions:
    enabled: bool
    mode: str
    template_vocab: Path
    gold_labels_train: Path
    gold_labels_valid: Path
    gold_labels_test: Path
    pseudo_labels_by_split: dict[str, Path]
    assignment_report: Path | None
    use_pseudo_splits: tuple[str, ...]
    lambda_gold: float
    lambda_pseudo: float
    head_hidden_dim: int | None
    dropout: float
    unknown_template_policy: str
    allow_gold_pseudo_overlap: bool
    allow_test_pseudo: bool
    max_pseudo_per_template: int | None
    control_seed: int


@dataclass(frozen=True)
class TemplateSupervision:
    pair_id: str
    has_gold_template: bool
    gold_template_id: str
    gold_template_index: int
    has_pseudo_template: bool
    pseudo_template_id: str
    pseudo_template_index: int
    template_supervision_source: str


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value


def _mapping_get(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    return getattr(config, key, default)


def _as_path(value: str | Path, *, name: str) -> Path:
    if value is None:
        raise ValueError(f"{name} is required")
    path = Path(value).expanduser()
    if not str(path):
        raise ValueError(f"{name} must be non-empty")
    return path


def _parse_group_by(value: Any) -> str | None:
    if value in (None, "", "none", "None"):
        return None
    return str(value)


def _parse_splits(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("train", "valid", "test")
    if isinstance(value, str):
        splits = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        splits = tuple(str(part).strip() for part in value if str(part).strip())
    allowed = {"train", "valid", "test"}
    unknown = sorted(set(splits) - allowed)
    if unknown:
        raise ValueError(f"final_splits contains unknown splits: {unknown}")
    if not splits:
        raise ValueError("final_splits must contain at least one split")
    return splits


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "none", "None", "null", "Null"):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def pairwise_options_from_config(config: Mapping[str, Any]) -> PairwiseTrainingOptions:
    pairwise = _mapping_get(config, "pairwise_training", {}) or {}
    runtime = _mapping_get(config, "runtime", {}) or {}
    pairwise_eval = _mapping_get(runtime, "pairwise_eval", {}) or {}

    split_dir = _mapping_get(pairwise, "split_dir", _mapping_get(pairwise_eval, "split_dir"))
    if split_dir is None:
        raise ValueError("pairwise_training.split_dir is required")

    relation_name = str(_mapping_get(pairwise, "relation_name", _mapping_get(pairwise_eval, "relation_name", "")))
    if not relation_name:
        raise ValueError("pairwise_training.relation_name is required")

    train_batch_size = int(_mapping_get(pairwise, "train_batch_size", _mapping_get(pairwise, "batch_size", 4)))
    eval_batch_size = int(_mapping_get(pairwise, "eval_batch_size", _mapping_get(pairwise_eval, "batch_size", 16)))
    validation_interval = int(_mapping_get(pairwise, "validation_interval", _mapping_get(runtime, "validation_interval", 3)))
    progress_log_interval = int(_mapping_get(pairwise, "progress_log_interval", _mapping_get(runtime, "progress_log_interval", 100)))
    early_stop_patience = _mapping_get(pairwise, "early_stop_patience", _mapping_get(runtime, "early_stop_patience"))
    if early_stop_patience in ("", "none", "None"):
        early_stop_patience = None

    k_values = tuple(int(value) for value in _mapping_get(pairwise, "k_values", _mapping_get(pairwise_eval, "k_values", [1, 5, 10])))
    if train_batch_size <= 0:
        raise ValueError(f"train_batch_size must be positive, got {train_batch_size}")
    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}")
    if validation_interval <= 0:
        raise ValueError(f"validation_interval must be positive, got {validation_interval}")
    if progress_log_interval < 0:
        raise ValueError(f"progress_log_interval must be non-negative, got {progress_log_interval}")
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError(f"k_values must be positive, got {k_values}")

    return PairwiseTrainingOptions(
        split_dir=_as_path(split_dir, name="pairwise_training.split_dir"),
        relation_name=relation_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        validation_interval=validation_interval,
        selection_metric=str(_mapping_get(pairwise, "selection_metric", _mapping_get(pairwise_eval, "selection_metric", "auprc"))),
        group_by=_parse_group_by(_mapping_get(pairwise, "group_by", _mapping_get(pairwise_eval, "group_by", "endpoint_id"))),
        k_values=k_values,
        progress_bar=bool(_mapping_get(pairwise, "progress_bar", _mapping_get(runtime, "progress_bar", True))),
        progress_log_interval=progress_log_interval,
        early_stop_patience=int(early_stop_patience) if early_stop_patience is not None else None,
        early_stop_min_delta=float(_mapping_get(pairwise, "early_stop_min_delta", _mapping_get(runtime, "early_stop_min_delta", 0.0))),
        shuffle=bool(_mapping_get(pairwise, "shuffle", True)),
        final_splits=_parse_splits(_mapping_get(pairwise, "final_splits", ("train", "valid", "test"))),
    )


def template_options_from_config(config: Mapping[str, Any]) -> TemplateTrainingOptions | None:
    template = _mapping_get(config, "template_training", {}) or {}
    if not bool(_mapping_get(template, "enabled", False)):
        return None

    mode = str(_mapping_get(template, "mode", "gold_pseudo"))
    allowed_modes = {"gold", "gold_pseudo", "shuffled_template", "random_pseudo"}
    if mode not in allowed_modes:
        raise ValueError(f"template_training.mode must be one of {sorted(allowed_modes)}, got {mode!r}")

    lambda_gold = float(_mapping_get(template, "lambda_gold", 1.0))
    lambda_pseudo = float(_mapping_get(template, "lambda_pseudo", 0.0 if mode == "gold" else 0.1))
    if lambda_gold < 0:
        raise ValueError(f"template_training.lambda_gold must be non-negative, got {lambda_gold}")
    if lambda_pseudo < 0:
        raise ValueError(f"template_training.lambda_pseudo must be non-negative, got {lambda_pseudo}")

    use_pseudo_splits = tuple(str(split) for split in _mapping_get(template, "use_pseudo_splits", ["train"]))
    unknown_splits = sorted(set(use_pseudo_splits) - {"train", "valid", "test"})
    if unknown_splits:
        raise ValueError(f"template_training.use_pseudo_splits contains unknown splits: {unknown_splits}")
    allow_test_pseudo = bool(_mapping_get(template, "allow_test_pseudo", False))
    if "test" in use_pseudo_splits and not allow_test_pseudo:
        raise ValueError("template_training.use_pseudo_splits cannot include test unless allow_test_pseudo=true")

    pseudo_labels_by_split = {}
    for split in use_pseudo_splits:
        key = f"pseudo_labels_{split}"
        value = _mapping_get(template, key)
        if value:
            pseudo_labels_by_split[split] = _as_path(value, name=f"template_training.{key}")

    if mode in {"gold_pseudo", "random_pseudo"} and "train" not in pseudo_labels_by_split:
        raise ValueError(f"template_training.mode={mode!r} requires template_training.pseudo_labels_train")

    assignment_report = _mapping_get(template, "assignment_report")
    return TemplateTrainingOptions(
        enabled=True,
        mode=mode,
        template_vocab=_as_path(_mapping_get(template, "template_vocab"), name="template_training.template_vocab"),
        gold_labels_train=_as_path(_mapping_get(template, "gold_labels_train"), name="template_training.gold_labels_train"),
        gold_labels_valid=_as_path(_mapping_get(template, "gold_labels_valid"), name="template_training.gold_labels_valid"),
        gold_labels_test=_as_path(_mapping_get(template, "gold_labels_test"), name="template_training.gold_labels_test"),
        pseudo_labels_by_split=pseudo_labels_by_split,
        assignment_report=_as_path(assignment_report, name="template_training.assignment_report") if assignment_report else None,
        use_pseudo_splits=use_pseudo_splits,
        lambda_gold=lambda_gold,
        lambda_pseudo=lambda_pseudo,
        head_hidden_dim=_optional_int(_mapping_get(template, "head_hidden_dim")),
        dropout=float(_mapping_get(template, "dropout", 0.0)),
        unknown_template_policy=str(_mapping_get(template, "unknown_template_policy", "error")),
        allow_gold_pseudo_overlap=bool(_mapping_get(template, "allow_gold_pseudo_overlap", False)),
        allow_test_pseudo=allow_test_pseudo,
        max_pseudo_per_template=_optional_int(_mapping_get(template, "max_pseudo_per_template")),
        control_seed=int(_mapping_get(template, "control_seed", 42)),
    )


def batch_records(
    records: Sequence[PairRecord],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[list[PairRecord]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    ordered = list(records)
    if shuffle:
        rng = random.Random(seed + epoch)
        rng.shuffle(ordered)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def _validate_template_label_metadata(
    record: PairRecord,
    *,
    label_pair_id: str,
    label_drug_id: str,
    label_endpoint_id: str,
    label_split: str,
    label_type: str,
) -> None:
    if label_pair_id != record.pair_id:
        raise ValueError(f"{label_type} label pair_id mismatch for pair {record.pair_id!r}")
    if label_drug_id != record.drug_id or label_endpoint_id != record.endpoint_id:
        raise ValueError(
            f"{label_type} label metadata does not match pair {record.pair_id!r}: "
            f"label has ({label_drug_id}, {label_endpoint_id}), pair has ({record.drug_id}, {record.endpoint_id})"
        )
    if label_split != record.split:
        raise ValueError(
            f"{label_type} label for pair {record.pair_id!r} has split {label_split!r}, "
            f"pair split is {record.split!r}"
        )


def _template_index_or_ignore(
    template_id: str,
    *,
    template_vocab: TemplateVocab,
    unknown_template_policy: str,
    pair_id: str,
    label_type: str,
) -> int:
    if template_id in template_vocab.template_id_to_index:
        return template_vocab.index(template_id)
    if unknown_template_policy == "ignore":
        return IGNORE_TEMPLATE_INDEX
    if unknown_template_policy != "error":
        raise ValueError(
            f"Unsupported unknown_template_policy {unknown_template_policy!r}; expected 'error' or 'ignore'"
        )
    raise ValueError(f"{label_type} template label for pair {pair_id!r} references unknown template ID {template_id!r}")


def _verify_assignment_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    checks = report.get("leakage_checks")
    if not isinstance(checks, dict):
        raise ValueError(f"{path} is missing leakage_checks")
    required_zero_keys = [
        "number_of_assigned_test_pairs",
        "number_of_assigned_valid_pairs",
        "number_of_assigned_negative_pairs",
        "number_of_assigned_gold_template_pairs",
    ]
    present = [key for key in required_zero_keys if key in checks]
    failures = {key: checks[key] for key in present if int(checks[key]) != 0}
    if failures:
        raise ValueError(f"Pseudo assignment leakage checks must be zero before training, got {failures}")
    return report


def _load_gold_labels_by_split(options: TemplateTrainingOptions) -> dict[str, dict[str, GoldTemplateLabelRow]]:
    return {
        "train": read_gold_template_label_tsv(options.gold_labels_train),
        "valid": read_gold_template_label_tsv(options.gold_labels_valid),
        "test": read_gold_template_label_tsv(options.gold_labels_test),
    }


def _load_pseudo_labels_by_split(options: TemplateTrainingOptions) -> dict[str, dict[str, PseudoTemplateLabelRow]]:
    labels = {}
    for split, path in options.pseudo_labels_by_split.items():
        labels[split] = cap_pseudo_labels_by_template(
            read_pseudo_template_label_tsv(path),
            max_per_template=options.max_pseudo_per_template,
        )
    return labels


def _shuffle_train_template_indices(
    supervision_by_split: dict[str, dict[str, TemplateSupervision]],
    *,
    seed: int,
) -> dict[str, dict[str, TemplateSupervision]]:
    shuffled = {split: dict(values) for split, values in supervision_by_split.items()}
    train_supervision = shuffled["train"]
    gold_items = [
        (pair_id, supervision.gold_template_index, supervision.gold_template_id)
        for pair_id, supervision in train_supervision.items()
        if supervision.has_gold_template
    ]
    pseudo_items = [
        (pair_id, supervision.pseudo_template_index, supervision.pseudo_template_id)
        for pair_id, supervision in train_supervision.items()
        if supervision.has_pseudo_template
    ]
    rng = random.Random(seed)
    for items, attr_prefix in ((gold_items, "gold"), (pseudo_items, "pseudo")):
        values = [(index, template_id) for _, index, template_id in items]
        rng.shuffle(values)
        for (pair_id, _, _), (new_index, new_template_id) in zip(items, values):
            supervision = train_supervision[pair_id]
            if attr_prefix == "gold":
                train_supervision[pair_id] = TemplateSupervision(
                    pair_id=supervision.pair_id,
                    has_gold_template=supervision.has_gold_template,
                    gold_template_id=new_template_id,
                    gold_template_index=new_index,
                    has_pseudo_template=supervision.has_pseudo_template,
                    pseudo_template_id=supervision.pseudo_template_id,
                    pseudo_template_index=supervision.pseudo_template_index,
                    template_supervision_source=supervision.template_supervision_source,
                )
            else:
                train_supervision[pair_id] = TemplateSupervision(
                    pair_id=supervision.pair_id,
                    has_gold_template=supervision.has_gold_template,
                    gold_template_id=supervision.gold_template_id,
                    gold_template_index=supervision.gold_template_index,
                    has_pseudo_template=supervision.has_pseudo_template,
                    pseudo_template_id=new_template_id,
                    pseudo_template_index=new_index,
                    template_supervision_source=supervision.template_supervision_source,
                )
    return shuffled


def _randomize_train_pseudo_indices(
    supervision_by_split: dict[str, dict[str, TemplateSupervision]],
    *,
    template_vocab: TemplateVocab,
    seed: int,
) -> dict[str, dict[str, TemplateSupervision]]:
    randomized = {split: dict(values) for split, values in supervision_by_split.items()}
    rng = random.Random(seed)
    template_ids = sorted(template_vocab.template_id_to_index)
    for pair_id, supervision in list(randomized["train"].items()):
        if not supervision.has_pseudo_template:
            continue
        template_id = rng.choice(template_ids)
        randomized["train"][pair_id] = TemplateSupervision(
            pair_id=supervision.pair_id,
            has_gold_template=supervision.has_gold_template,
            gold_template_id=supervision.gold_template_id,
            gold_template_index=supervision.gold_template_index,
            has_pseudo_template=True,
            pseudo_template_id=template_id,
            pseudo_template_index=template_vocab.index(template_id),
            template_supervision_source="random_pseudo",
        )
    return randomized


def build_template_supervision(
    split_records: dict[str, Sequence[PairRecord]],
    *,
    options: TemplateTrainingOptions,
    template_vocab: TemplateVocab,
) -> dict[str, dict[str, TemplateSupervision]]:
    gold_by_split = _load_gold_labels_by_split(options)
    pseudo_by_split = _load_pseudo_labels_by_split(options)
    _verify_assignment_report(options.assignment_report)

    supervision_by_split: dict[str, dict[str, TemplateSupervision]] = {}
    for split, records in split_records.items():
        pair_ids = {record.pair_id for record in records}
        extra_gold = sorted(set(gold_by_split[split]) - pair_ids)
        if extra_gold:
            raise ValueError(f"Gold-template labels for split {split!r} include unknown pair_ids: {extra_gold[:10]}")
        pseudo_labels = pseudo_by_split.get(split, {})
        extra_pseudo = sorted(set(pseudo_labels) - pair_ids)
        if extra_pseudo:
            raise ValueError(f"Pseudo-template labels for split {split!r} include unknown pair_ids: {extra_pseudo[:10]}")

        split_supervision = {}
        for record in records:
            gold_label = gold_by_split[split].get(record.pair_id)
            pseudo_label = pseudo_labels.get(record.pair_id)

            has_gold = False
            gold_template_id = ""
            gold_template_index = IGNORE_TEMPLATE_INDEX
            if gold_label is not None:
                _validate_template_label_metadata(
                    record,
                    label_pair_id=gold_label.pair_id,
                    label_drug_id=gold_label.drug_id,
                    label_endpoint_id=gold_label.endpoint_id,
                    label_split=gold_label.split,
                    label_type="gold-template",
                )
                gold_template_index = _template_index_or_ignore(
                    gold_label.primary_template_id,
                    template_vocab=template_vocab,
                    unknown_template_policy=options.unknown_template_policy,
                    pair_id=record.pair_id,
                    label_type="gold",
                )
                has_gold = record.label == 1 and gold_template_index != IGNORE_TEMPLATE_INDEX
                gold_template_id = gold_label.primary_template_id if has_gold else ""

            has_pseudo = False
            pseudo_template_id = ""
            pseudo_template_index = IGNORE_TEMPLATE_INDEX
            if pseudo_label is not None:
                _validate_template_label_metadata(
                    record,
                    label_pair_id=pseudo_label.pair_id,
                    label_drug_id=pseudo_label.drug_id,
                    label_endpoint_id=pseudo_label.endpoint_id,
                    label_split=pseudo_label.split,
                    label_type="pseudo-template",
                )
                if split == "test" and not options.allow_test_pseudo:
                    raise ValueError("Test pseudo-template labels are rejected by default")
                if has_gold and not options.allow_gold_pseudo_overlap:
                    raise ValueError(
                        f"Pair {record.pair_id!r} has both gold-template and pseudo-template labels; "
                        "pseudo supervision must target no-gold positive pairs"
                    )
                pseudo_template_index = _template_index_or_ignore(
                    pseudo_label.pseudo_template_id,
                    template_vocab=template_vocab,
                    unknown_template_policy=options.unknown_template_policy,
                    pair_id=record.pair_id,
                    label_type="pseudo",
                )
                has_pseudo = (
                    record.label == 1
                    and not has_gold
                    and split in options.use_pseudo_splits
                    and pseudo_template_index != IGNORE_TEMPLATE_INDEX
                )
                pseudo_template_id = pseudo_label.pseudo_template_id if has_pseudo else ""

            if options.mode == "gold":
                has_pseudo = False
                pseudo_template_id = ""
                pseudo_template_index = IGNORE_TEMPLATE_INDEX

            if has_gold:
                source = "gold"
            elif has_pseudo:
                source = "pseudo"
            else:
                source = "none"

            split_supervision[record.pair_id] = TemplateSupervision(
                pair_id=record.pair_id,
                has_gold_template=has_gold,
                gold_template_id=gold_template_id,
                gold_template_index=gold_template_index if has_gold else IGNORE_TEMPLATE_INDEX,
                has_pseudo_template=has_pseudo,
                pseudo_template_id=pseudo_template_id,
                pseudo_template_index=pseudo_template_index if has_pseudo else IGNORE_TEMPLATE_INDEX,
                template_supervision_source=source,
            )
        supervision_by_split[split] = split_supervision

    if options.mode == "shuffled_template":
        supervision_by_split = _shuffle_train_template_indices(
            supervision_by_split,
            seed=options.control_seed,
        )
    elif options.mode == "random_pseudo":
        supervision_by_split = _randomize_train_pseudo_indices(
            supervision_by_split,
            template_vocab=template_vocab,
            seed=options.control_seed,
        )
    return supervision_by_split


def summarize_template_supervision(
    supervision_by_split: dict[str, dict[str, TemplateSupervision]],
) -> dict[str, dict[str, int]]:
    summary = {}
    for split, rows in supervision_by_split.items():
        values = list(rows.values())
        summary[split] = {
            "num_examples": len(values),
            "num_gold_template_examples": sum(1 for item in values if item.has_gold_template),
            "num_pseudo_template_examples": sum(1 for item in values if item.has_pseudo_template),
            "num_template_supervised_examples": sum(
                1 for item in values if item.has_gold_template or item.has_pseudo_template
            ),
        }
    return summary


def should_validate_epoch(epoch: int, *, num_epoch: int, validation_interval: int) -> bool:
    if epoch <= 0 or num_epoch <= 0:
        return False
    return epoch % validation_interval == 0 or epoch == num_epoch


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_checkpoint(
    solver,
    path: str | Path,
    *,
    epoch: int,
    metrics: dict[str, Any] | None = None,
    template_head=None,
) -> None:
    import torch

    path = Path(path)
    state = {
        "model": solver.model.state_dict(),
        "optimizer": solver.optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
    }
    if template_head is not None:
        state["template_head"] = template_head.state_dict()
    torch.save(state, path)


def _load_checkpoint_for_pairwise_training(solver, checkpoint: str | Path, *, template_head=None) -> None:
    import torch

    _solver_load_without_graphs(solver, str(checkpoint), load_optimizer=False)
    if template_head is None:
        return
    try:
        state = torch.load(checkpoint, map_location=solver.device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=solver.device)
    if "template_head" not in state:
        raise ValueError(f"Checkpoint {checkpoint} does not contain template_head state")
    template_head.load_state_dict(state["template_head"])


def _validate_relation_and_entities(records: Sequence[PairRecord], *, dataset, relation_name: str, split: str) -> int:
    if relation_name not in dataset.inv_relation_vocab:
        raise ValueError(f"Relation {relation_name!r} is missing from BioPathNet relation vocabulary")
    missing = []
    for record in records:
        if record.drug_id not in dataset.inv_entity_vocab:
            missing.append((record.pair_id, "drug_id", record.drug_id))
        if record.endpoint_id not in dataset.inv_entity_vocab:
            missing.append((record.pair_id, "endpoint_id", record.endpoint_id))
        if len(missing) >= 10:
            break
    if missing:
        raise ValueError(f"Split {split!r} contains pairs with entities missing from BioPathNet graph: {missing}")
    return int(dataset.inv_relation_vocab[relation_name])


def _pairwise_tensors(records: Sequence[PairRecord], *, dataset, solver, relation_index: int):
    import torch

    device = solver.device
    h_index = torch.tensor(
        [dataset.inv_entity_vocab[record.drug_id] for record in records],
        dtype=torch.long,
        device=device,
    ).unsqueeze(-1)
    t_index = torch.tensor(
        [dataset.inv_entity_vocab[record.endpoint_id] for record in records],
        dtype=torch.long,
        device=device,
    ).unsqueeze(-1)
    r_index = torch.full((len(records), 1), relation_index, dtype=torch.long, device=device)
    return h_index, t_index, r_index


def _pairwise_logits(records: Sequence[PairRecord], *, dataset, solver, relation_index: int):
    h_index, t_index, r_index = _pairwise_tensors(
        records,
        dataset=dataset,
        solver=solver,
        relation_index=relation_index,
    )
    return solver.model.model(solver.model.fact_graph, h_index, t_index, r_index).view(-1)


def _pairwise_logits_and_features(records: Sequence[PairRecord], *, dataset, solver, relation_index: int):
    """Return BioPathNet pair logits and the MLP-input pair representation.

    This mirrors the original NBFNet forward pass without changing
    biopathnet/original. The representation is the path-aware feature gathered
    for the queried drug-endpoint pair immediately before the final scoring MLP.
    """

    import torch

    model = solver.model.model
    graph = solver.model.fact_graph
    h_index, t_index, r_index = _pairwise_tensors(
        records,
        dataset=dataset,
        solver=solver,
        relation_index=relation_index,
    )

    if not all(hasattr(model, name) for name in ("bellmanford", "negative_sample_to_tail", "mlp")):
        logits = model(graph, h_index, t_index, r_index).view(-1)
        return logits, logits.unsqueeze(-1)

    shape = h_index.shape
    if graph.num_relation:
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = model.negative_sample_to_tail(h_index, t_index, r_index)
    else:
        graph = model.as_relational_graph(graph)
        h_index = h_index.view(-1, 1)
        t_index = t_index.view(-1, 1)
        r_index = torch.zeros_like(h_index)

    output = model.bellmanford(graph, h_index[:, 0], r_index[:, 0])
    feature = output["node_feature"].transpose(0, 1)
    index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
    feature = feature.gather(1, index)

    if getattr(model, "symmetric", False):
        r_index = (r_index + model.num_relation) % (model.num_relation * 2)
        output = model.bellmanford(graph, t_index[:, 0], r_index[:, 0])
        inv_feature = output["node_feature"].transpose(0, 1)
        index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
        inv_feature = inv_feature.gather(1, index)
        feature = (feature + inv_feature) / 2

    logits = model.mlp(feature).squeeze(-1).reshape(shape).reshape(-1)
    return logits, feature.reshape(-1, feature.shape[-1])


def _infer_pair_representation_dim(solver) -> int:
    import torch

    mlp = getattr(solver.model.model, "mlp", None)
    if mlp is not None:
        for module in mlp.modules():
            if isinstance(module, torch.nn.Linear):
                return int(module.in_features)
    raise ValueError("Cannot infer BioPathNet pair representation dimension from the model MLP")


def _train_one_epoch(
    *,
    records: Sequence[PairRecord],
    dataset,
    solver,
    options: PairwiseTrainingOptions,
    epoch: int,
    num_epoch: int,
    seed: int,
    logger: logging.Logger,
    template_head=None,
    template_options: TemplateTrainingOptions | None = None,
    template_supervision: Mapping[str, TemplateSupervision] | None = None,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    if template_head is not None:
        from mechrep.models.losses import compute_semi_template_loss

    relation_index = _validate_relation_and_entities(
        records,
        dataset=dataset,
        relation_name=options.relation_name,
        split="train",
    )
    batches = batch_records(
        records,
        batch_size=options.train_batch_size,
        shuffle=options.shuffle,
        seed=seed,
        epoch=epoch,
    )
    iterator = iter_progress(
        enumerate(batches, start=1),
        enabled=options.progress_bar,
        total=len(batches),
        desc=f"BioPathNet pairwise epoch {epoch}",
        unit="batch",
    )

    solver.model.train()
    if template_head is not None:
        template_head.train()
    solver.model.split = "train"
    total_loss = 0.0
    total_loss_pred = 0.0
    total_loss_gold = 0.0
    total_loss_pseudo = 0.0
    total_examples = 0
    positive_score_sum = 0.0
    negative_score_sum = 0.0
    num_positive = 0
    num_negative = 0
    num_gold_template_examples = 0
    num_pseudo_template_examples = 0

    for batch_id, batch in iterator:
        labels = torch.tensor([record.label for record in batch], dtype=torch.float, device=solver.device)
        solver.optimizer.zero_grad()
        if template_head is None:
            logits = _pairwise_logits(batch, dataset=dataset, solver=solver, relation_index=relation_index)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss_pred = loss
            loss_gold = logits.sum() * 0.0
            loss_pseudo = logits.sum() * 0.0
        else:
            if template_options is None or template_supervision is None:
                raise ValueError("template_head training requires template_options and template_supervision")
            logits, pair_features = _pairwise_logits_and_features(
                batch,
                dataset=dataset,
                solver=solver,
                relation_index=relation_index,
            )
            template_logits = template_head(pair_features)
            gold_index = torch.tensor(
                [template_supervision[record.pair_id].gold_template_index for record in batch],
                dtype=torch.long,
                device=solver.device,
            )
            pseudo_index = torch.tensor(
                [template_supervision[record.pair_id].pseudo_template_index for record in batch],
                dtype=torch.long,
                device=solver.device,
            )
            loss_parts = compute_semi_template_loss(
                logits,
                labels,
                template_logits,
                gold_index,
                pseudo_index,
                lambda_gold=template_options.lambda_gold,
                lambda_pseudo=template_options.lambda_pseudo,
            )
            loss = loss_parts["loss_total"]
            loss_pred = loss_parts["loss_pred"]
            loss_gold = loss_parts["loss_gold"]
            loss_pseudo = loss_parts["loss_pseudo"]
            num_gold_template_examples += int(loss_parts["num_gold_template_examples"])
            num_pseudo_template_examples += int(loss_parts["num_pseudo_template_examples"])
        loss.backward()
        solver.optimizer.step()

        batch_size = len(batch)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_loss_pred += float(loss_pred.detach().cpu()) * batch_size
        total_loss_gold += float(loss_gold.detach().cpu()) * batch_size
        total_loss_pseudo += float(loss_pseudo.detach().cpu()) * batch_size
        total_examples += batch_size
        with torch.no_grad():
            scores = torch.sigmoid(logits.detach())
            positive_mask = labels == 1
            negative_mask = labels == 0
            if bool(positive_mask.any()):
                positive_score_sum += float(scores[positive_mask].sum().detach().cpu())
                num_positive += int(positive_mask.sum().detach().cpu())
            if bool(negative_mask.any()):
                negative_score_sum += float(scores[negative_mask].sum().detach().cpu())
                num_negative += int(negative_mask.sum().detach().cpu())

        if (
            options.progress_log_interval
            and (
                batch_id == 1
                or batch_id % options.progress_log_interval == 0
                or batch_id == len(batches)
            )
        ):
            logger.warning(
                "BioPathNet pairwise progress: epoch %d/%d batch %d/%d (%.1f%%) loss %.5g",
                epoch,
                num_epoch,
                batch_id,
                len(batches),
                batch_id / len(batches) * 100,
                float(loss.detach().cpu()),
            )
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": float(loss.detach().cpu())})

    return {
        "epoch": float(epoch),
        "loss": total_loss / total_examples,
        "loss_pred": total_loss_pred / total_examples,
        "loss_gold": total_loss_gold / total_examples,
        "loss_pseudo": total_loss_pseudo / total_examples,
        "num_examples": float(total_examples),
        "num_positive": float(num_positive),
        "num_negative": float(num_negative),
        "num_gold_template_examples": float(num_gold_template_examples),
        "num_pseudo_template_examples": float(num_pseudo_template_examples),
        "mean_positive_score": positive_score_sum / num_positive if num_positive else math.nan,
        "mean_negative_score": negative_score_sum / num_negative if num_negative else math.nan,
    }


def _evaluate_pairwise_split(
    *,
    records: Sequence[PairRecord],
    dataset,
    solver,
    options: PairwiseTrainingOptions,
    split: str,
    output_dir: Path,
    suffix: str,
    template_head=None,
    template_vocab: TemplateVocab | None = None,
    template_supervision: Mapping[str, TemplateSupervision] | None = None,
) -> dict[str, Any]:
    solver.model.eval()
    if template_head is not None:
        if template_vocab is None or template_supervision is None:
            raise ValueError("Template evaluation requires template_vocab and template_supervision")
        template_head.eval()
        rows = _score_records_with_template(
            records,
            dataset=dataset,
            solver=solver,
            relation_name=options.relation_name,
            split=split,
            batch_size=options.eval_batch_size,
            progress_bar=options.progress_bar,
            template_head=template_head,
            template_vocab=template_vocab,
            template_supervision=template_supervision,
        )
    else:
        rows = score_records(
            records,
            dataset=dataset,
            solver=solver,
            relation_name=options.relation_name,
            split=split,
            batch_size=options.eval_batch_size,
            progress_bar=options.progress_bar,
        )
    metrics = evaluate_predictions(rows, k_values=options.k_values, group_by=options.group_by)
    metrics["num_examples"] = len(rows)
    metrics["num_positive"] = sum(int(row["label"]) for row in rows)
    metrics["num_negative"] = len(rows) - metrics["num_positive"]
    if template_supervision is not None:
        supervised = [template_supervision[row["pair_id"]] for row in rows]
        metrics["template_supervision"] = {
            "num_gold_template_examples": sum(1 for item in supervised if item.has_gold_template),
            "num_pseudo_template_examples": sum(1 for item in supervised if item.has_pseudo_template),
            "num_template_supervised_examples": sum(
                1 for item in supervised if item.has_gold_template or item.has_pseudo_template
            ),
        }

    predictions_path = output_dir / f"predictions_{split}_{suffix}.tsv"
    metrics_path = output_dir / f"metrics_{split}_{suffix}.json"
    _write_prediction_rows(predictions_path, rows)
    write_metrics_json(metrics_path, metrics)
    return {
        "split": split,
        "suffix": suffix,
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "metrics_summary": metrics,
    }


def _write_prediction_rows(path: str | Path, rows: Sequence[dict]) -> None:
    if not rows:
        _write_predictions(path, rows)
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _score_records_with_template(
    records: Sequence[PairRecord],
    *,
    dataset,
    solver,
    relation_name: str,
    split: str,
    batch_size: int,
    progress_bar: bool,
    template_head,
    template_vocab: TemplateVocab,
    template_supervision: Mapping[str, TemplateSupervision],
) -> list[dict]:
    import torch

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    relation_index = _validate_relation_and_entities(records, dataset=dataset, relation_name=relation_name, split=split)

    rows = []
    iterator = range(0, len(records), batch_size)
    iterator = iter_progress(iterator, enabled=progress_bar, desc=f"score {split}", unit="batch")
    with torch.no_grad():
        for start in iterator:
            batch = records[start : start + batch_size]
            logits, pair_features = _pairwise_logits_and_features(
                batch,
                dataset=dataset,
                solver=solver,
                relation_index=relation_index,
            )
            scores = torch.sigmoid(logits.detach()).cpu().tolist()
            template_logits = template_head(pair_features)
            template_probs = torch.softmax(template_logits, dim=-1)
            top_k = min(3, template_vocab.size)
            top_confidence, top_indices = torch.topk(template_probs, k=top_k, dim=-1)
            for row_index, (record, score) in enumerate(zip(batch, scores)):
                supervision = template_supervision[record.pair_id]
                top_template_ids = [
                    template_vocab.template_id(int(index.item())) for index in top_indices[row_index]
                ]
                rows.append(
                    {
                        "pair_id": record.pair_id,
                        "drug_id": record.drug_id,
                        "endpoint_id": record.endpoint_id,
                        "label": str(record.label),
                        "score": f"{float(score):.8f}",
                        "split": split,
                        "has_gold_template": "1" if supervision.has_gold_template else "0",
                        "gold_template_id": supervision.gold_template_id,
                        "has_pseudo_template": "1" if supervision.has_pseudo_template else "0",
                        "pseudo_template_id": supervision.pseudo_template_id,
                        "template_supervision_source": supervision.template_supervision_source,
                        "predicted_template_id": top_template_ids[0],
                        "template_confidence": f"{float(top_confidence[row_index, 0].item()):.8f}",
                        "template_top3_ids": "|".join(top_template_ids),
                    }
                )
    return rows


def _load_original_solver(config_path: Path, *, seed: int):
    import numpy as np
    import torch
    from torchdrug import core, models as _torchdrug_models  # noqa: F401
    from torchdrug.utils import comm

    from biopathnet import dataset as _dataset  # noqa: F401
    from biopathnet import layer as _layer  # noqa: F401
    from biopathnet import model as _model  # noqa: F401
    from biopathnet import task as _task  # noqa: F401
    from biopathnet import util

    args_seed = int(seed)
    seed_rank = args_seed + int(comm.get_rank())
    random.seed(args_seed)
    np.random.seed(args_seed)
    torch.manual_seed(seed_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_rank)

    cfg = util.load_config(str(config_path), context={})
    working_dir = Path(util.create_working_directory(cfg)).resolve()
    logger = util.get_root_logger()
    logger.warning("Working directory: %s", working_dir)
    logger.warning("Input Seed: %d", args_seed)
    logger.warning("Set Seed: %d", seed_rank)
    if comm.get_rank() == 0:
        logger.warning("Config file: %s", config_path)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    return cfg, dataset, solver, logger, working_dir


def run_pairwise_training(config_path: str | Path, *, seed: int = 42) -> dict[str, Any]:
    root = repository_root()
    prepare_environment(root)
    config_path = Path(config_path).resolve(strict=False)
    cfg, dataset, solver, logger, working_dir = _load_original_solver(config_path, seed=seed)
    options = pairwise_options_from_config(cfg)
    template_options = template_options_from_config(cfg)

    split_records = {
        split: read_pair_tsv(options.split_dir / f"{split}.tsv", split=split)
        for split in ("train", "valid", "test")
    }
    for split, records in split_records.items():
        _validate_relation_and_entities(records, dataset=dataset, relation_name=options.relation_name, split=split)

    template_vocab = None
    template_head = None
    template_supervision = None
    template_summary = None
    if template_options is not None:
        from mechrep.models.template_head import TemplateHead

        template_vocab = TemplateVocab.load(template_options.template_vocab)
        template_supervision = build_template_supervision(
            split_records,
            options=template_options,
            template_vocab=template_vocab,
        )
        template_summary = summarize_template_supervision(template_supervision)
        pair_representation_dim = _infer_pair_representation_dim(solver)
        template_head = TemplateHead(
            input_dim=pair_representation_dim,
            num_templates=template_vocab.size,
            hidden_dim=template_options.head_hidden_dim,
            dropout=template_options.dropout,
        ).to(solver.device)
        solver.optimizer.add_param_group({"params": list(template_head.parameters())})
        template_vocab.save(working_dir / "template_vocab.json")

    shutil.copy2(config_path, working_dir / "config.yaml")
    _write_json(
        working_dir / "seed.json",
        {
            "seed": int(seed),
            "relation_name": options.relation_name,
            "split_dir": str(options.split_dir),
            "template_training": _to_plain(template_options) if template_options is not None else None,
        },
    )
    if template_summary is not None:
        _write_json(working_dir / "template_supervision_summary.json", template_summary)

    num_epoch = int(cfg.train.num_epoch)
    if num_epoch <= 0:
        raise ValueError(f"train.num_epoch must be positive, got {num_epoch}")

    history = []
    validation_dir = working_dir / "pairwise_validation"
    final_dir = working_dir / "pairwise_final"
    checkpoints = []
    early_stopper = EarlyStopper(
        options.selection_metric,
        options.early_stop_patience,
        min_delta=options.early_stop_min_delta,
    )
    best_checkpoint: Path | None = None
    best_epoch = 0
    best_metric_value: float | None = None

    logger.warning("Explicit pairwise training options: %s", pprint.pformat(options))
    if template_options is not None:
        logger.warning("Template training options: %s", pprint.pformat(template_options))
        logger.warning("Template supervision summary: %s", pprint.pformat(template_summary))
    for epoch in range(1, num_epoch + 1):
        train_metrics = _train_one_epoch(
            records=split_records["train"],
            dataset=dataset,
            solver=solver,
            options=options,
            epoch=epoch,
            num_epoch=num_epoch,
            seed=seed,
            logger=logger,
            template_head=template_head,
            template_options=template_options,
            template_supervision=template_supervision["train"] if template_supervision is not None else None,
        )
        checkpoint_path = working_dir / f"model_epoch_{epoch}.pth"
        _save_checkpoint(
            solver,
            checkpoint_path,
            epoch=epoch,
            metrics={"train": train_metrics},
            template_head=template_head,
        )
        checkpoints.append(str(checkpoint_path))

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "train": train_metrics,
        }

        if should_validate_epoch(epoch, num_epoch=num_epoch, validation_interval=options.validation_interval):
            valid_report = _evaluate_pairwise_split(
                records=split_records["valid"],
                dataset=dataset,
                solver=solver,
                options=options,
                split="valid",
                output_dir=validation_dir,
                suffix=f"epoch_{epoch}",
                template_head=template_head,
                template_vocab=template_vocab,
                template_supervision=template_supervision["valid"] if template_supervision is not None else None,
            )
            valid_metrics = valid_report["metrics_summary"]
            epoch_record["valid"] = valid_report
            if options.selection_metric not in valid_metrics:
                raise ValueError(f"selection_metric {options.selection_metric!r} is unavailable")
            metric_value = float(valid_metrics[options.selection_metric])
            improved, should_stop = early_stopper.observe(metric_value, epoch=epoch)
            if improved or best_checkpoint is None:
                best_checkpoint = checkpoint_path
                best_epoch = epoch
                best_metric_value = metric_value
            logger.warning(
                "Explicit pairwise valid epoch %d: %s %.5g AUROC %.5g AUPRC %.5g",
                epoch,
                options.selection_metric,
                metric_value,
                valid_metrics["auroc"],
                valid_metrics["auprc"],
            )
            if should_stop:
                epoch_record["early_stop"] = True
                history.append(epoch_record)
                logger.warning(
                    "Explicit pairwise early stopping at epoch %d; best epoch %d",
                    epoch,
                    early_stopper.best_epoch,
                )
                break
        else:
            if best_checkpoint is None:
                best_checkpoint = checkpoint_path
                best_epoch = epoch

        history.append(epoch_record)
        _write_json(working_dir / "history.json", history)

    if best_checkpoint is None:
        best_checkpoint = Path(checkpoints[-1])
        best_epoch = int(history[-1]["epoch"])

    _load_checkpoint_for_pairwise_training(solver, best_checkpoint, template_head=template_head)
    final_reports = {}
    for split in options.final_splits:
        final_reports[split] = _evaluate_pairwise_split(
            records=split_records[split],
            dataset=dataset,
            solver=solver,
            options=options,
            split=split,
            output_dir=final_dir,
            suffix=f"best_epoch_{best_epoch}",
            template_head=template_head,
            template_vocab=template_vocab,
            template_supervision=template_supervision[split] if template_supervision is not None else None,
        )

    summary = {
        "working_dir": str(working_dir),
        "config": str(config_path),
        "seed": int(seed),
        "relation_name": options.relation_name,
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": int(best_epoch),
        "best_selection_metric": options.selection_metric,
        "best_selection_metric_value": best_metric_value,
        "history": history,
        "final": final_reports,
        "checkpoints": checkpoints,
        "template_training": _to_plain(template_options) if template_options is not None else None,
        "template_supervision_summary": template_summary,
    }
    _write_json(working_dir / "summary.json", summary)
    _write_json(
        working_dir / "metrics.json",
        {
            split: report["metrics_summary"]
            for split, report in final_reports.items()
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-s", "--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pairwise_training(args.config, seed=args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
