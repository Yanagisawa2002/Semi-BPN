"""Score endpoint-OOD pair files with a trained original BioPathNet checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Sequence

import yaml

from mechrep.data.build_pairs import PairRecord, read_pair_tsv
from mechrep.evaluation.eval_prediction import evaluate_predictions, write_metrics_json
from mechrep.training.progress import iter_progress
from mechrep.training.run_original_biopathnet_linux import prepare_environment, repository_root, _solver_load_without_graphs


PREDICTION_COLUMNS = ("pair_id", "drug_id", "endpoint_id", "label", "score", "split")


def load_yaml_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def find_latest_checkpoint(search_dir: str | Path, pattern: str = "model_epoch_*.pth") -> Path:
    search_dir = Path(search_dir)
    candidates = sorted(search_dir.rglob(pattern), key=lambda path: (path.stat().st_mtime, str(path)))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching {pattern!r} found under {search_dir}")
    return candidates[-1]


def infer_relation_name(dataset_dir: str | Path, train_file: str = "train2.txt") -> str:
    path = Path(dataset_dir) / train_file
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"{path} contains a non-triplet row: {row}")
            return row[1]
    raise ValueError(f"Cannot infer relation name from empty file: {path}")


def validate_records_mappable(records: Sequence[PairRecord], entity_vocab: dict[str, int], *, split: str) -> None:
    missing = []
    for record in records:
        if record.drug_id not in entity_vocab:
            missing.append((record.pair_id, "drug_id", record.drug_id))
        if record.endpoint_id not in entity_vocab:
            missing.append((record.pair_id, "endpoint_id", record.endpoint_id))
        if len(missing) >= 10:
            break
    if missing:
        raise ValueError(f"Split {split!r} contains pairs with entities missing from BioPathNet graph: {missing}")


def _write_predictions(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_solver(config_path: Path, checkpoint: Path):
    import numpy as np
    import torch
    from torchdrug import core, models as _torchdrug_models  # noqa: F401

    from biopathnet import dataset as _dataset  # noqa: F401
    from biopathnet import layer as _layer  # noqa: F401
    from biopathnet import model as _model  # noqa: F401
    from biopathnet import task as _task  # noqa: F401
    from biopathnet import util

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = util.load_config(str(config_path), context={})
    cfg.pop("fast_test", None)
    cfg.pop("checkpoint", None)
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    _solver_load_without_graphs(solver, str(checkpoint), load_optimizer=False)
    solver.model.eval()
    return cfg, dataset, solver


def score_records(
    records: Sequence[PairRecord],
    *,
    dataset,
    solver,
    relation_name: str,
    split: str,
    batch_size: int,
    progress_bar: bool,
) -> list[dict]:
    import torch

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if relation_name not in dataset.inv_relation_vocab:
        raise ValueError(f"Relation {relation_name!r} is missing from BioPathNet relation vocabulary")

    validate_records_mappable(records, dataset.inv_entity_vocab, split=split)
    relation_index = dataset.inv_relation_vocab[relation_name]
    device = solver.device
    task = solver.model
    graph = task.fact_graph

    rows = []
    iterator = range(0, len(records), batch_size)
    iterator = iter_progress(iterator, enabled=progress_bar, desc=f"score {split}", unit="batch")
    with torch.no_grad():
        for start in iterator:
            batch = records[start : start + batch_size]
            h_index = torch.tensor(
                [dataset.inv_entity_vocab[record.drug_id] for record in batch],
                dtype=torch.long,
                device=device,
            ).unsqueeze(-1)
            t_index = torch.tensor(
                [dataset.inv_entity_vocab[record.endpoint_id] for record in batch],
                dtype=torch.long,
                device=device,
            ).unsqueeze(-1)
            r_index = torch.full((len(batch), 1), relation_index, dtype=torch.long, device=device)
            logits = task.model(graph, h_index, t_index, r_index)
            scores = torch.sigmoid(logits.squeeze(-1)).detach().cpu().tolist()
            for record, score in zip(batch, scores):
                rows.append(
                    {
                        "pair_id": record.pair_id,
                        "drug_id": record.drug_id,
                        "endpoint_id": record.endpoint_id,
                        "label": str(record.label),
                        "score": f"{float(score):.8f}",
                        "split": split,
                    }
                )
    return rows


def run_pair_scoring(
    *,
    config_path: str | Path,
    checkpoint: str | Path | None,
    split_dir: str | Path,
    output_dir: str | Path | None,
    relation_name: str | None,
    splits: Sequence[str],
    batch_size: int,
    k_values: Sequence[int],
    group_by: str | None,
    progress_bar: bool,
) -> dict:
    root = repository_root()
    prepare_environment(root)
    config_path = Path(config_path).resolve(strict=False)
    config = load_yaml_config(config_path)
    dataset_dir = Path(config["dataset"]["path"])
    checkpoint_path = Path(checkpoint).resolve(strict=False) if checkpoint else find_latest_checkpoint(config["output_dir"])
    output_path = Path(output_dir) if output_dir else checkpoint_path.parent / "pairwise_eval"
    output_path.mkdir(parents=True, exist_ok=True)
    relation = relation_name or infer_relation_name(dataset_dir)

    _, dataset, solver = _load_solver(config_path, checkpoint_path)

    split_dir = Path(split_dir)
    result = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "output_dir": str(output_path),
        "relation_name": relation,
        "splits": {},
    }
    for split in splits:
        records = read_pair_tsv(split_dir / f"{split}.tsv", split=split)
        rows = score_records(
            records,
            dataset=dataset,
            solver=solver,
            relation_name=relation,
            split=split,
            batch_size=batch_size,
            progress_bar=progress_bar,
        )
        prediction_path = output_path / f"predictions_{split}.tsv"
        metrics_path = output_path / f"metrics_{split}.json"
        _write_predictions(prediction_path, rows)
        metrics = evaluate_predictions(rows, k_values=k_values, group_by=group_by)
        metrics["num_examples"] = len(rows)
        metrics["num_positive"] = sum(int(row["label"]) for row in rows)
        metrics["num_negative"] = len(rows) - metrics["num_positive"]
        write_metrics_json(metrics_path, metrics)
        result["splits"][split] = {
            "predictions": str(prediction_path),
            "metrics": str(metrics_path),
            "metrics_summary": metrics,
        }

    with (output_path / "pairwise_eval_config.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--relation-name")
    parser.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, nargs="*", default=[10, 50, 100])
    parser.add_argument("--group-by", default="endpoint_id")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_by = args.group_by
    if group_by in ("", "none", "None"):
        group_by = None
    result = run_pair_scoring(
        config_path=args.config,
        checkpoint=args.checkpoint,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        relation_name=args.relation_name,
        splits=args.splits,
        batch_size=args.batch_size,
        k_values=args.k,
        group_by=group_by,
        progress_bar=not args.no_progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
