"""Train endpoint pair prediction with supervised gold-template learning."""

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
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise RuntimeError("Gold-template training requires PyTorch") from exc

from mechrep.data.gold_template_dataset import (
    GoldTemplateExample,
    GoldTemplatePairDataset,
)
from mechrep.evaluation.eval_gold_template import evaluate_gold_template_predictions
from mechrep.evaluation.eval_prediction import write_metrics_json
from mechrep.models.biopathnet_wrapper import PairEmbeddingBioPathNetWrapper
from mechrep.models.losses import joint_gold_template_loss
from mechrep.models.template_head import TemplateHead
from mechrep.templates.extract_templates import make_template_id
from mechrep.templates.template_vocab import TemplateVocab, build_template_vocab
from mechrep.training.progress import EarlyStopper, iter_progress, normalize_monitor_metric


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
    "template_top3_ids",
)


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


def _write_tsv(path: str | Path, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def create_toy_gold_template_inputs(config: dict) -> None:
    data_config = config["data"]
    template_config = config["templates"]
    split_dir = Path(data_config["split_dir"])
    split_rows = {
        "train": [
            ("P_train_1", "D1", "E_train", "1"),
            ("P_train_0", "D2", "E_train", "0"),
            ("P_train_2", "D3", "E_train2", "1"),
            ("P_train_3", "D4", "E_train2", "0"),
        ],
        "valid": [
            ("P_valid_1", "D1", "E_valid", "1"),
            ("P_valid_0", "D2", "E_valid", "0"),
        ],
        "test": [
            ("P_test_1", "D1", "E_test", "1"),
            ("P_test_0", "D2", "E_test", "0"),
        ],
    }
    for split, rows in split_rows.items():
        _write_tsv(split_dir / f"{split}.tsv", ["pair_id", "drug_id", "endpoint_id", "label"], rows)

    template_a = make_template_id("Drug|Gene|Endpoint", "targets|associated_with")
    template_b = make_template_id("Drug|Pathway|Endpoint", "participates_in|associated_with")
    template_table = Path(template_config["template_table"])
    _write_tsv(
        template_table,
        ["template_id", "node_type_sequence", "relation_type_sequence", "support_count", "pair_ids", "path_ids"],
        [
            (template_a, "Drug|Gene|Endpoint", "targets|associated_with", 2, "P_train_1|P_valid_1", "path_1|path_2"),
            (template_b, "Drug|Pathway|Endpoint", "participates_in|associated_with", 1, "P_train_2", "path_3"),
        ],
    )
    label_columns = [
        "pair_id",
        "drug_id",
        "endpoint_id",
        "split",
        "template_ids",
        "primary_template_id",
        "num_gold_paths",
    ]
    for split, rows in {
        "train": [
            ("P_train_1", "D1", "E_train", "train", template_a, template_a, 1),
            ("P_train_2", "D3", "E_train2", "train", template_b, template_b, 1),
        ],
        "valid": [("P_valid_1", "D1", "E_valid", "valid", template_a, template_a, 1)],
        "test": [("P_test_1", "D1", "E_test", "test", template_a, template_a, 1)],
    }.items():
        _write_tsv(template_config[f"{split}_labels"], label_columns, rows)


def ensure_inputs(config: dict, *, create_toy_data_if_missing: bool) -> None:
    data_config = config["data"]
    template_config = config["templates"]
    required = [
        Path(data_config["split_dir"]) / "train.tsv",
        Path(data_config["split_dir"]) / "valid.tsv",
        Path(data_config["split_dir"]) / "test.tsv",
        Path(template_config["template_table"]),
        Path(template_config["train_labels"]),
        Path(template_config["valid_labels"]),
        Path(template_config["test_labels"]),
    ]
    if all(path.exists() for path in required):
        return
    if create_toy_data_if_missing or config.get("toy_data", {}).get("create_if_missing", False):
        create_toy_gold_template_inputs(config)
        return
    missing = [str(path) for path in required if not path.exists()]
    raise FileNotFoundError(f"Missing required gold-template training inputs: {missing}")


def build_vocab(values: Sequence[str]) -> Dict[str, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}


def examples_to_tensors(
    examples: Sequence[GoldTemplateExample],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    drug_index = torch.tensor([drug_to_index[example.drug_id] for example in examples], dtype=torch.long)
    endpoint_index = torch.tensor([endpoint_to_index[example.endpoint_id] for example in examples], dtype=torch.long)
    pair_label = torch.tensor([example.label for example in examples], dtype=torch.float32)
    template_index = torch.tensor([example.primary_template_index for example in examples], dtype=torch.long)
    return drug_index, endpoint_index, pair_label, template_index


def batches(examples: Sequence[GoldTemplateExample], *, batch_size: int, seed: int, epoch: int) -> List[List[GoldTemplateExample]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    order = list(range(len(examples)))
    random.Random(seed + epoch).shuffle(order)
    return [[examples[index] for index in order[start : start + batch_size]] for start in range(0, len(order), batch_size)]


def train_one_model(
    encoder: PairEmbeddingBioPathNetWrapper,
    template_head: TemplateHead,
    train_examples: Sequence[GoldTemplateExample],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    lambda_gold: float,
    seed: int,
    valid_examples: Sequence[GoldTemplateExample] | None = None,
    template_vocab: TemplateVocab | None = None,
    k_values: Sequence[int] = (10, 50, 100),
    group_by: str | None = "endpoint_id",
    early_stopping_metric: str | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
    progress_bar: bool = True,
) -> List[dict]:
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(template_head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = []
    stopper = EarlyStopper(early_stopping_metric, patience, min_delta=min_delta)
    monitor_key = normalize_monitor_metric(early_stopping_metric or "")
    best_state = None

    epoch_iter = iter_progress(range(epochs), enabled=progress_bar, desc="gold-template train", unit="epoch")
    for epoch in epoch_iter:
        encoder.train()
        template_head.train()
        totals = {
            "loss_pred": 0.0,
            "loss_gold": 0.0,
            "loss_total": 0.0,
            "num_examples": 0,
            "num_gold_template_examples": 0,
        }
        epoch_batches = batches(train_examples, batch_size=batch_size, seed=seed, epoch=epoch)
        batch_iter = iter_progress(
            epoch_batches,
            enabled=progress_bar,
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )
        for batch in batch_iter:
            drug_index, endpoint_index, pair_label, template_index = examples_to_tensors(
                batch, drug_to_index, endpoint_to_index
            )
            outputs = encoder(drug_index, endpoint_index)
            template_logits = template_head(outputs["pair_representation"])
            loss_parts = joint_gold_template_loss(
                outputs["pair_score"],
                pair_label,
                template_logits,
                template_index,
                lambda_gold=lambda_gold,
            )
            optimizer.zero_grad()
            loss_parts["loss_total"].backward()
            optimizer.step()

            batch_size_actual = len(batch)
            totals["loss_pred"] += float(loss_parts["loss_pred"].detach()) * batch_size_actual
            totals["loss_gold"] += float(loss_parts["loss_gold"].detach()) * batch_size_actual
            totals["loss_total"] += float(loss_parts["loss_total"].detach()) * batch_size_actual
            totals["num_examples"] += batch_size_actual
            totals["num_gold_template_examples"] += loss_parts["num_gold_template_examples_in_batch"]
        epoch_record = {
            "epoch": epoch + 1,
            "loss_pred": totals["loss_pred"] / totals["num_examples"],
            "loss_gold": totals["loss_gold"] / totals["num_examples"],
            "loss_total": totals["loss_total"] / totals["num_examples"],
            "num_gold_template_examples": totals["num_gold_template_examples"],
        }
        if stopper.enabled:
            if valid_examples is None or template_vocab is None:
                raise ValueError("early stopping requires valid_examples and template_vocab")
            valid_rows = predict_examples(
                encoder,
                template_head,
                valid_examples,
                drug_to_index,
                endpoint_to_index,
                template_vocab,
            )
            valid_metrics = evaluate_gold_template_predictions(
                valid_rows,
                k_values=k_values,
                group_by=group_by,
                num_templates=template_vocab.size,
            )
            if monitor_key not in valid_metrics:
                raise ValueError(f"early stopping metric {monitor_key!r} is not available in validation metrics")
            monitor_value = float(valid_metrics[monitor_key])
            improved, should_stop = stopper.observe(monitor_value, epoch=epoch + 1)
            epoch_record[early_stopping_metric or monitor_key] = monitor_value
            epoch_record["early_stop_bad_epochs"] = stopper.bad_epochs
            if improved:
                best_state = {
                    "encoder": copy.deepcopy(encoder.state_dict()),
                    "template_head": copy.deepcopy(template_head.state_dict()),
                }
            if should_stop:
                epoch_record["early_stopped"] = True
                history.append(epoch_record)
                break
        history.append(epoch_record)

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        template_head.load_state_dict(best_state["template_head"])
    return history


def predict_examples(
    encoder: PairEmbeddingBioPathNetWrapper,
    template_head: TemplateHead,
    examples: Sequence[GoldTemplateExample],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
    template_vocab: TemplateVocab,
) -> List[dict]:
    encoder.eval()
    template_head.eval()
    with torch.no_grad():
        drug_index, endpoint_index, _, _ = examples_to_tensors(examples, drug_to_index, endpoint_to_index)
        outputs = encoder(drug_index, endpoint_index)
        pair_scores = torch.sigmoid(outputs["pair_score"])
        template_logits = template_head(outputs["pair_representation"])
        template_probs = torch.softmax(template_logits, dim=-1)
        top_k = min(3, template_vocab.size)
        top_confidence, top_indices = torch.topk(template_probs, k=top_k, dim=-1)

    rows = []
    for row_index, example in enumerate(examples):
        predicted_index = int(top_indices[row_index, 0].item())
        predicted_template_id = template_vocab.template_id(predicted_index)
        top_template_ids = [template_vocab.template_id(int(index.item())) for index in top_indices[row_index]]
        rows.append(
            {
                "pair_id": example.pair_id,
                "drug_id": example.drug_id,
                "endpoint_id": example.endpoint_id,
                "label": str(example.label),
                "score": f"{float(pair_scores[row_index].item()):.8f}",
                "split": example.split,
                "has_gold_template": "1" if example.has_gold_template else "0",
                "primary_template_id": example.primary_template_id,
                "predicted_template_id": predicted_template_id,
                "template_confidence": f"{float(top_confidence[row_index, 0].item()):.8f}",
                "template_top3_ids": "|".join(top_template_ids),
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


def run_gold_template_training(config: dict, *, create_toy_data_if_missing: bool = False) -> dict:
    experiment_config = config.get("experiment", {})
    seed = int(experiment_config.get("seed", config.get("seed", 0)))
    set_seed(seed)
    output_dir = Path(experiment_config.get("output_dir", config.get("output_dir", "results/gold_template")))
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_inputs(config, create_toy_data_if_missing=create_toy_data_if_missing)

    template_config = config["templates"]
    template_vocab = build_template_vocab(
        template_config["template_table"],
        train_labels=template_config.get("train_labels"),
        use_train_templates_only=bool(template_config.get("use_train_templates_only", True)),
    )
    template_vocab_path = output_dir / "template_vocab.json"
    template_vocab.save(template_vocab_path)

    dataset = GoldTemplatePairDataset.from_files(
        split_dir=config["data"]["split_dir"],
        train_labels=template_config["train_labels"],
        valid_labels=template_config["valid_labels"],
        test_labels=template_config["test_labels"],
        template_vocab=template_vocab,
        unknown_template_policy=template_config.get("unknown_template_policy", "error"),
    )
    all_examples = list(dataset.all_examples())
    drug_to_index = build_vocab([example.drug_id for example in all_examples])
    endpoint_to_index = build_vocab([example.endpoint_id for example in all_examples])

    model_config = config.get("model", {})
    embedding_dim = int(model_config.get("embedding_dim", model_config.get("hidden_dim", 64)))
    encoder = PairEmbeddingBioPathNetWrapper(
        num_drugs=len(drug_to_index),
        num_endpoints=len(endpoint_to_index),
        embedding_dim=embedding_dim,
    )
    template_head = TemplateHead(
        encoder.representation_dim,
        template_vocab.size,
        hidden_dim=int(model_config.get("template_hidden_dim", model_config.get("hidden_dim", 0))) or None,
        dropout=float(model_config.get("dropout", 0.0)),
    )

    eval_config = config.get("evaluation", {})
    k_values = [int(k) for k in eval_config.get("k_values", [10, 50, 100])]
    group_by = eval_config.get("group_by", "endpoint_id")
    if group_by in ("", "none", "None", None):
        group_by = None

    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    history = train_one_model(
        encoder,
        template_head,
        dataset.examples("train"),
        drug_to_index,
        endpoint_to_index,
        epochs=int(training_config.get("epochs", 50)),
        batch_size=int(training_config.get("batch_size", 128)),
        learning_rate=float(training_config.get("learning_rate", 0.0001)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        lambda_gold=float(loss_config.get("lambda_gold", 1.0)),
        seed=seed,
        valid_examples=dataset.examples("valid"),
        template_vocab=template_vocab,
        k_values=k_values,
        group_by=group_by,
        early_stopping_metric=training_config.get("early_stopping_metric"),
        patience=training_config.get("patience"),
        min_delta=float(training_config.get("min_delta", 0.0)),
        progress_bar=bool(training_config.get("progress_bar", True)),
    )

    prediction_rows = {}
    metrics = {}
    for split in ("train", "valid", "test"):
        examples = dataset.examples(split)
        prediction_rows[split] = predict_examples(
            encoder,
            template_head,
            examples,
            drug_to_index,
            endpoint_to_index,
            template_vocab,
        )
        metrics[split] = evaluate_gold_template_predictions(
            prediction_rows[split],
            k_values=k_values,
            group_by=group_by,
            num_templates=template_vocab.size,
        )
        metrics[split]["num_examples"] = len(examples)
        metrics[split]["num_templates"] = template_vocab.size
        if split == "train" and history:
            metrics[split]["final_loss_pred"] = history[-1]["loss_pred"]
            metrics[split]["final_loss_gold"] = history[-1]["loss_gold"]
            metrics[split]["final_loss_total"] = history[-1]["loss_total"]
        write_predictions(output_dir / f"predictions_{split}.tsv", prediction_rows[split])
        write_metrics_json(output_dir / f"metrics_{split}.json", metrics[split])

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "template_head_state": template_head.state_dict(),
            "drug_to_index": drug_to_index,
            "endpoint_to_index": endpoint_to_index,
            "template_vocab": template_vocab.as_json_dict(),
            "seed": seed,
            "backend": "pair_embedding_wrapper",
        },
        checkpoint_path,
    )

    run_config = dict(config)
    run_config["resolved"] = {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "template_vocab": str(template_vocab_path),
        "num_templates": template_vocab.size,
        "backend": "pair_embedding_wrapper",
        "pair_representation": "concat(drug_embedding, endpoint_embedding, product, abs_difference)",
    }
    save_config(output_dir / "config.yaml", run_config)
    with (output_dir / "random_seed.json").open("w", encoding="utf-8") as handle:
        json.dump({"seed": seed}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "template_vocab_path": str(template_vocab_path),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gold_template.yaml")
    parser.add_argument("--create-toy-data-if-missing", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--lambda-gold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.lambda_gold is not None:
        config.setdefault("loss", {})["lambda_gold"] = args.lambda_gold
    result = run_gold_template_training(config, create_toy_data_if_missing=args.create_toy_data_if_missing)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
