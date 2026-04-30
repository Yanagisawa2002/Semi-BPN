"""Minimal endpoint-level baseline training entrypoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise RuntimeError("The baseline training entrypoint requires PyTorch") from exc

from mechrep.data.biopathnet_adapter import DEFAULT_RELATION_NAME, EndpointBioPathNetAdapter
from mechrep.data.build_ood_splits import build_ood_splits
from mechrep.data.build_pairs import PairRecord
from mechrep.evaluation.eval_prediction import evaluate_predictions, write_metrics_json
from mechrep.training.progress import EarlyStopper, iter_progress, normalize_monitor_metric


PREDICTION_COLUMNS = ("pair_id", "drug_id", "endpoint_id", "label", "score", "split")


class PairEmbeddingBaseline(nn.Module):
    """Small deterministic pair scorer used when TorchDrug BioPathNet is unavailable."""

    def __init__(self, num_drugs: int, num_endpoints: int, embedding_dim: int):
        super().__init__()
        self.drug_embedding = nn.Embedding(num_drugs, embedding_dim)
        self.endpoint_embedding = nn.Embedding(num_endpoints, embedding_dim)
        self.drug_bias = nn.Embedding(num_drugs, 1)
        self.endpoint_bias = nn.Embedding(num_endpoints, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, drug_index: torch.Tensor, endpoint_index: torch.Tensor) -> torch.Tensor:
        drug = self.drug_embedding(drug_index)
        endpoint = self.endpoint_embedding(endpoint_index)
        dot = (drug * endpoint).sum(dim=-1)
        drug_bias = self.drug_bias(drug_index).squeeze(-1)
        endpoint_bias = self.endpoint_bias(endpoint_index).squeeze(-1)
        return dot + drug_bias + endpoint_bias + self.global_bias


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


def save_config(path: str | Path, config: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _toy_pair_rows() -> List[tuple[str, str, str, str]]:
    rows = []
    for endpoint_index in range(6):
        endpoint = f"toy_endpoint_{endpoint_index}"
        positive_drug = endpoint_index % 3
        for drug_index in range(3):
            label = "1" if drug_index == positive_drug else "0"
            rows.append((f"toy_pair_{endpoint_index}_{drug_index}", f"toy_drug_{drug_index}", endpoint, label))
    return rows


def create_toy_splits(split_dir: str | Path, *, seed: int) -> None:
    split_dir = Path(split_dir)
    source_path = split_dir.parent / "toy_pairs.tsv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["pair_id", "drug_id", "endpoint_id", "label"])
        writer.writerows(_toy_pair_rows())
    build_ood_splits(
        source_path,
        split_dir,
        split_type="endpoint_ood",
        seed=seed,
        valid_ratio=1 / 3,
        test_ratio=1 / 3,
    )


def ensure_split_dir(config: dict, *, create_toy_data_if_missing: bool) -> Path:
    split_dir = Path(config["data"]["split_dir"])
    required = [split_dir / f"{split}.tsv" for split in ("train", "valid", "test")]
    if all(path.exists() for path in required):
        return split_dir
    if create_toy_data_if_missing or config.get("toy_data", {}).get("create_if_missing", False):
        create_toy_splits(split_dir, seed=int(config.get("seed", 0)))
        return split_dir
    missing = [str(path) for path in required if not path.exists()]
    raise FileNotFoundError(f"Missing split files: {missing}")


def inspect_kg(config: dict) -> dict:
    kg_config = config.get("kg", {})
    path_value = kg_config.get("path")
    if not path_value:
        return {"path": None, "exists": False, "header": []}
    path = Path(path_value)
    info = {"path": str(path), "exists": path.exists(), "header": []}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            first_line = handle.readline().strip()
        info["header"] = first_line.split(",") if first_line else []
    elif kg_config.get("required", False):
        raise FileNotFoundError(f"Configured KG path does not exist: {path}")
    return info


def build_vocab(values: Sequence[str]) -> Dict[str, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}


def records_to_tensors(
    records: Sequence[PairRecord],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drug_index = torch.tensor([drug_to_index[record.drug_id] for record in records], dtype=torch.long)
    endpoint_index = torch.tensor([endpoint_to_index[record.endpoint_id] for record in records], dtype=torch.long)
    labels = torch.tensor([record.label for record in records], dtype=torch.float32)
    return drug_index, endpoint_index, labels


def predict_records(
    model: PairEmbeddingBaseline,
    records: Sequence[PairRecord],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
) -> List[dict]:
    model.eval()
    with torch.no_grad():
        drug_index, endpoint_index, _ = records_to_tensors(records, drug_to_index, endpoint_to_index)
        scores = torch.sigmoid(model(drug_index, endpoint_index)).tolist()
    rows = []
    for record, score in zip(records, scores):
        rows.append(
            {
                "pair_id": record.pair_id,
                "drug_id": record.drug_id,
                "endpoint_id": record.endpoint_id,
                "label": str(record.label),
                "score": f"{score:.8f}",
                "split": record.split or "",
            }
        )
    return rows


def write_predictions(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train_model(
    model: PairEmbeddingBaseline,
    train_records: Sequence[PairRecord],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    valid_records: Sequence[PairRecord] | None = None,
    k_values: Sequence[int] = (10, 50, 100),
    group_by: str | None = "endpoint_id",
    early_stopping_metric: str | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
    progress_bar: bool = True,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    drug_index, endpoint_index, labels = records_to_tensors(train_records, drug_to_index, endpoint_to_index)
    losses = []
    stopper = EarlyStopper(early_stopping_metric, patience, min_delta=min_delta)
    monitor_key = normalize_monitor_metric(early_stopping_metric or "")
    best_state = None
    best_epoch = None
    best_loss = None

    for epoch in iter_progress(range(epochs), enabled=progress_bar, desc="baseline train", unit="epoch"):
        model.train()
        optimizer.zero_grad()
        logits = model(drug_index, endpoint_index)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if stopper.enabled:
            if valid_records is None:
                raise ValueError("early stopping requires valid_records")
            _, valid_metrics = evaluate_split(
                model,
                valid_records,
                drug_to_index,
                endpoint_to_index,
                k_values=k_values,
                group_by=group_by,
            )
            if monitor_key not in valid_metrics:
                raise ValueError(f"early stopping metric {monitor_key!r} is not available in validation metrics")
            improved, should_stop = stopper.observe(float(valid_metrics[monitor_key]), epoch=epoch + 1)
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                best_loss = losses[-1]
            if should_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    selected_epoch = best_epoch if best_epoch is not None else len(losses)
    selected_loss = best_loss if best_loss is not None else (losses[-1] if losses else None)
    return {
        "losses": losses,
        "selected_epoch": selected_epoch,
        "selected_loss": selected_loss,
    }


def evaluate_split(
    model: PairEmbeddingBaseline,
    records: Sequence[PairRecord],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
    *,
    k_values: Sequence[int],
    group_by: str | None,
) -> tuple[List[dict], dict]:
    rows = predict_records(model, records, drug_to_index, endpoint_to_index)
    return rows, evaluate_predictions(rows, k_values=k_values, group_by=group_by)


def run_baseline(config: dict, *, create_toy_data_if_missing: bool = False) -> dict:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    output_dir = Path(config.get("output_dir", "results/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = ensure_split_dir(config, create_toy_data_if_missing=create_toy_data_if_missing)
    kg_info = inspect_kg(config)

    relation_name = config.get("adapter", {}).get("relation_name", DEFAULT_RELATION_NAME)
    adapter = EndpointBioPathNetAdapter.from_split_dir(split_dir, relation_name=relation_name)
    adapter_counts = adapter.write_adapter_outputs(output_dir / "adapter")

    all_records = list(adapter.records("train") + adapter.records("valid") + adapter.records("test"))
    drug_to_index = build_vocab([record.drug_id for record in all_records])
    endpoint_to_index = build_vocab([record.endpoint_id for record in all_records])

    model_config = config.get("model", {})
    model = PairEmbeddingBaseline(
        num_drugs=len(drug_to_index),
        num_endpoints=len(endpoint_to_index),
        embedding_dim=int(model_config.get("embedding_dim", 16)),
    )
    training_config = config.get("training", {})
    eval_config = config.get("evaluation", {})
    k_values = [int(k) for k in eval_config.get("k_values", [10, 50, 100])]
    group_by = eval_config.get("group_by", "endpoint_id")
    if group_by in ("", "none", "None", None):
        group_by = None

    training_result = train_model(
        model,
        adapter.records("train"),
        drug_to_index,
        endpoint_to_index,
        epochs=int(training_config.get("epochs", 50)),
        learning_rate=float(training_config.get("learning_rate", 0.05)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        valid_records=adapter.records("valid"),
        k_values=k_values,
        group_by=group_by,
        early_stopping_metric=training_config.get("early_stopping_metric"),
        patience=training_config.get("patience"),
        min_delta=float(training_config.get("min_delta", 0.0)),
        progress_bar=bool(training_config.get("progress_bar", True)),
    )
    losses = training_result["losses"]

    prediction_rows = {}
    metrics = {}
    for split in ("train", "valid", "test"):
        prediction_rows[split], metrics[split] = evaluate_split(
            model,
            adapter.records(split),
            drug_to_index,
            endpoint_to_index,
            k_values=k_values,
            group_by=group_by,
        )
        metrics[split]["num_examples"] = len(adapter.records(split))
        if split == "train":
            metrics[split]["final_loss"] = training_result["selected_loss"]
            metrics[split]["selected_epoch"] = training_result["selected_epoch"]
        write_metrics_json(output_dir / f"metrics_{split}.json", metrics[split])

    write_predictions(output_dir / "predictions_test.tsv", prediction_rows["test"])
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "drug_to_index": drug_to_index,
            "endpoint_to_index": endpoint_to_index,
            "relation_name": relation_name,
            "seed": seed,
            "backend": "pair_embedding_wrapper",
        },
        checkpoint_path,
    )

    run_config = dict(config)
    run_config["resolved"] = {
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "kg": kg_info,
        "adapter_positive_triple_counts": adapter_counts,
        "checkpoint_path": str(checkpoint_path),
        "backend": "pair_embedding_wrapper",
        "training": training_result,
    }
    save_config(output_dir / "config.yaml", run_config)
    with (output_dir / "random_seed.json").open("w", encoding="utf-8") as handle:
        json.dump({"seed": seed}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--create-toy-data-if-missing", action="store_true")
    parser.add_argument("--split-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.split_dir:
        config.setdefault("data", {})["split_dir"] = args.split_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    result = run_baseline(config, create_toy_data_if_missing=args.create_toy_data_if_missing)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
