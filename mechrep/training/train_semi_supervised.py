"""Train semi-supervised template-guided endpoint pair prediction."""

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
    raise RuntimeError("Semi-supervised template training requires PyTorch") from exc

from mechrep.data.semi_template_dataset import SemiTemplateExample, SemiTemplatePairDataset
from mechrep.evaluation.eval_prediction import write_metrics_json
from mechrep.evaluation.eval_semi_template import evaluate_semi_template_predictions
from mechrep.models.biopathnet_wrapper import PairEmbeddingBioPathNetWrapper
from mechrep.models.losses import compute_semi_template_loss
from mechrep.models.template_head import TemplateHead
from mechrep.templates.extract_templates import make_template_id
from mechrep.templates.template_vocab import TemplateVocab
from mechrep.training.progress import EarlyStopper, iter_progress, normalize_monitor_metric


PREDICTION_COLUMNS = (
    "pair_id",
    "drug_id",
    "endpoint_id",
    "label",
    "score",
    "split",
    "has_gold_template",
    "gold_template_id",
    "has_pseudo_template",
    "pseudo_template_id",
    "template_supervision_source",
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


def create_toy_semi_template_inputs(config: dict) -> None:
    output_dir = Path(config["experiment"]["output_dir"])
    toy_dir = output_dir / "toy_inputs"
    split_dir = toy_dir / "splits"
    template_dir = toy_dir / "templates"
    pseudo_dir = toy_dir / "pseudo"

    config["data"]["split_dir"] = str(split_dir)
    config["data"]["train_file"] = str(split_dir / "train.tsv")
    config["data"]["valid_file"] = str(split_dir / "valid.tsv")
    config["data"]["test_file"] = str(split_dir / "test.tsv")
    config["templates"]["template_table"] = str(template_dir / "templates.tsv")
    config["templates"]["template_vocab"] = str(template_dir / "template_vocab.json")
    config["templates"]["gold_labels_train"] = str(template_dir / "gold_template_labels_train.tsv")
    config["templates"]["gold_labels_valid"] = str(template_dir / "gold_template_labels_valid.tsv")
    config["templates"]["gold_labels_test"] = str(template_dir / "gold_template_labels_test.tsv")
    config["pseudo_templates"]["pseudo_labels_train"] = str(pseudo_dir / "pseudo_template_labels_train.tsv")
    config["pseudo_templates"]["assignment_report"] = str(pseudo_dir / "assignment_report.json")

    pair_columns = ["pair_id", "drug_id", "endpoint_id", "label"]
    _write_tsv(
        split_dir / "train.tsv",
        pair_columns,
        [
            ("P_gold", "D1", "E_train", 1),
            ("P_pseudo", "D2", "E_train", 1),
            ("P_none", "D3", "E_train", 1),
            ("P_neg", "D4", "E_train", 0),
        ],
    )
    _write_tsv(split_dir / "valid.tsv", pair_columns, [("P_valid_gold", "D1", "E_valid", 1), ("P_valid_neg", "D4", "E_valid", 0)])
    _write_tsv(split_dir / "test.tsv", pair_columns, [("P_test_gold", "D1", "E_test", 1), ("P_test_neg", "D4", "E_test", 0)])

    template_a = make_template_id("Drug|Gene|Endpoint", "targets|associated_with")
    template_b = make_template_id("Drug|Pathway|Endpoint", "participates_in|associated_with")
    _write_tsv(
        template_dir / "templates.tsv",
        ["template_id", "node_type_sequence", "relation_type_sequence", "support_count", "pair_ids", "path_ids"],
        [
            (template_a, "Drug|Gene|Endpoint", "targets|associated_with", 2, "P_gold|P_valid_gold", "path_a"),
            (template_b, "Drug|Pathway|Endpoint", "participates_in|associated_with", 1, "P_pseudo", "path_b"),
        ],
    )
    TemplateVocab.from_template_ids([template_a, template_b]).save(template_dir / "template_vocab.json")

    label_columns = ["pair_id", "drug_id", "endpoint_id", "split", "template_ids", "primary_template_id", "num_gold_paths"]
    _write_tsv(template_dir / "gold_template_labels_train.tsv", label_columns, [("P_gold", "D1", "E_train", "train", template_a, template_a, 1)])
    _write_tsv(template_dir / "gold_template_labels_valid.tsv", label_columns, [("P_valid_gold", "D1", "E_valid", "valid", template_a, template_a, 1)])
    _write_tsv(template_dir / "gold_template_labels_test.tsv", label_columns, [("P_test_gold", "D1", "E_test", "test", template_a, template_a, 1)])

    pseudo_columns = [
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
    ]
    _write_tsv(
        pseudo_dir / "pseudo_template_labels_train.tsv",
        pseudo_columns,
        [
            (
                "P_pseudo",
                "D2",
                "E_train",
                "train",
                template_b,
                1,
                "path_pseudo",
                0.95,
                1.0,
                1.0,
                0.98,
                "high_confidence_path_template_match",
            )
        ],
    )
    report = {
        "leakage_checks": {
            "number_of_assigned_test_pairs": 0,
            "number_of_assigned_valid_pairs": 0,
            "number_of_assigned_negative_pairs": 0,
            "number_of_assigned_gold_template_pairs": 0,
        },
        "assigned_no_gold_positive_pairs": 1,
    }
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    with (pseudo_dir / "assignment_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_inputs(config: dict, *, create_toy_data: bool) -> None:
    if create_toy_data or config.get("toy_data", {}).get("create_if_missing", False):
        create_toy_semi_template_inputs(config)
        return
    required = [
        Path(config["data"]["split_dir"]) / "train.tsv",
        Path(config["data"]["split_dir"]) / "valid.tsv",
        Path(config["data"]["split_dir"]) / "test.tsv",
        Path(config["templates"]["template_vocab"]),
        Path(config["templates"]["gold_labels_train"]),
        Path(config["templates"]["gold_labels_valid"]),
        Path(config["templates"]["gold_labels_test"]),
        Path(config["pseudo_templates"]["pseudo_labels_train"]),
        Path(config["pseudo_templates"]["assignment_report"]),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required semi-template training inputs: {missing}")


def verify_assignment_report(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    checks = report.get("leakage_checks")
    if not isinstance(checks, dict):
        raise ValueError(f"{path} is missing leakage_checks")
    required = [
        "number_of_assigned_test_pairs",
        "number_of_assigned_valid_pairs",
        "number_of_assigned_negative_pairs",
        "number_of_assigned_gold_template_pairs",
    ]
    missing = [key for key in required if key not in checks]
    if missing:
        raise ValueError(f"{path} leakage_checks is missing keys: {missing}")
    failures = {key: checks[key] for key in required if int(checks[key]) != 0}
    if failures:
        raise ValueError(f"Pseudo assignment leakage checks must all be zero before training, got {failures}")
    return report


def build_vocab(values: Sequence[str]) -> Dict[str, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}


def pseudo_label_paths(config: dict) -> Dict[str, str]:
    pseudo_config = config["pseudo_templates"]
    paths = {}
    for split in pseudo_config.get("use_pseudo_splits", ["train"]):
        key = f"pseudo_labels_{split}"
        if key not in pseudo_config:
            raise ValueError(f"Missing pseudo_templates.{key} for enabled pseudo split {split!r}")
        paths[split] = pseudo_config[key]
    return paths


def build_dataset(config: dict, template_vocab: TemplateVocab) -> SemiTemplatePairDataset:
    pseudo_config = config["pseudo_templates"]
    return SemiTemplatePairDataset.from_files(
        split_dir=config["data"]["split_dir"],
        gold_labels_train=config["templates"]["gold_labels_train"],
        gold_labels_valid=config["templates"]["gold_labels_valid"],
        gold_labels_test=config["templates"]["gold_labels_test"],
        pseudo_labels_by_split=pseudo_label_paths(config),
        template_vocab=template_vocab,
        use_pseudo_splits=pseudo_config.get("use_pseudo_splits", ["train"]),
        allow_gold_pseudo_overlap=bool(pseudo_config.get("allow_gold_pseudo_overlap", False)),
        allow_test_pseudo=bool(pseudo_config.get("allow_test_pseudo", False)),
        unknown_template_policy=config["templates"].get("unknown_template_policy", "error"),
        max_pseudo_per_template=pseudo_config.get("max_pseudo_per_template"),
    )


def examples_to_tensors(
    examples: Sequence[SemiTemplateExample],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    drug_index = torch.tensor([drug_to_index[example.drug_id] for example in examples], dtype=torch.long)
    endpoint_index = torch.tensor([endpoint_to_index[example.endpoint_id] for example in examples], dtype=torch.long)
    pair_label = torch.tensor([example.label for example in examples], dtype=torch.float32)
    gold_index = torch.tensor([example.gold_template_index for example in examples], dtype=torch.long)
    pseudo_index = torch.tensor([example.pseudo_template_index for example in examples], dtype=torch.long)
    return drug_index, endpoint_index, pair_label, gold_index, pseudo_index


def batches(examples: Sequence[SemiTemplateExample], *, batch_size: int, seed: int, epoch: int) -> List[List[SemiTemplateExample]]:
    order = list(range(len(examples)))
    random.Random(seed + epoch).shuffle(order)
    return [[examples[index] for index in order[start : start + batch_size]] for start in range(0, len(order), batch_size)]


def train_one_model(
    encoder: PairEmbeddingBioPathNetWrapper,
    template_head: TemplateHead,
    train_examples: Sequence[SemiTemplateExample],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    lambda_gold: float,
    lambda_pseudo: float,
    seed: int,
    valid_examples: Sequence[SemiTemplateExample] | None = None,
    template_vocab: TemplateVocab | None = None,
    k_values: Sequence[int] = (10, 50, 100),
    group_by: str | None = "endpoint_id",
    early_stopping_metric: str | None = None,
    patience: int | None = None,
    min_delta: float = 0.0,
    progress_bar: bool = True,
) -> List[dict]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(template_head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = []
    stopper = EarlyStopper(early_stopping_metric, patience, min_delta=min_delta)
    monitor_key = normalize_monitor_metric(early_stopping_metric or "")
    best_state = None

    epoch_iter = iter_progress(range(epochs), enabled=progress_bar, desc="semi-template train", unit="epoch")
    for epoch in epoch_iter:
        totals = {
            "loss_pred": 0.0,
            "loss_gold": 0.0,
            "loss_pseudo": 0.0,
            "loss_total": 0.0,
            "num_examples": 0,
            "num_gold_template_examples": 0,
            "num_pseudo_template_examples": 0,
        }
        encoder.train()
        template_head.train()
        epoch_batches = batches(train_examples, batch_size=batch_size, seed=seed, epoch=epoch)
        batch_iter = iter_progress(
            epoch_batches,
            enabled=progress_bar,
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )
        for batch in batch_iter:
            drug_index, endpoint_index, pair_label, gold_index, pseudo_index = examples_to_tensors(
                batch, drug_to_index, endpoint_to_index
            )
            outputs = encoder(drug_index, endpoint_index)
            template_logits = template_head(outputs["pair_representation"])
            loss_parts = compute_semi_template_loss(
                outputs["pair_score"],
                pair_label,
                template_logits,
                gold_index,
                pseudo_index,
                lambda_gold=lambda_gold,
                lambda_pseudo=lambda_pseudo,
            )
            optimizer.zero_grad()
            loss_parts["loss_total"].backward()
            optimizer.step()

            n = len(batch)
            totals["loss_pred"] += float(loss_parts["loss_pred"].detach()) * n
            totals["loss_gold"] += float(loss_parts["loss_gold"].detach()) * n
            totals["loss_pseudo"] += float(loss_parts["loss_pseudo"].detach()) * n
            totals["loss_total"] += float(loss_parts["loss_total"].detach()) * n
            totals["num_examples"] += n
            totals["num_gold_template_examples"] += loss_parts["num_gold_template_examples"]
            totals["num_pseudo_template_examples"] += loss_parts["num_pseudo_template_examples"]
        epoch_record = {
            "epoch": epoch + 1,
            "loss_pred": totals["loss_pred"] / totals["num_examples"],
            "loss_gold": totals["loss_gold"] / totals["num_examples"],
            "loss_pseudo": totals["loss_pseudo"] / totals["num_examples"],
            "loss_total": totals["loss_total"] / totals["num_examples"],
            "num_gold_template_examples": totals["num_gold_template_examples"],
            "num_pseudo_template_examples": totals["num_pseudo_template_examples"],
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
            valid_metrics = evaluate_semi_template_predictions(
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
    examples: Sequence[SemiTemplateExample],
    drug_to_index: Dict[str, int],
    endpoint_to_index: Dict[str, int],
    template_vocab: TemplateVocab,
) -> List[dict]:
    encoder.eval()
    template_head.eval()
    with torch.no_grad():
        drug_index, endpoint_index, _, _, _ = examples_to_tensors(examples, drug_to_index, endpoint_to_index)
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
                "gold_template_id": example.gold_template_id,
                "has_pseudo_template": "1" if example.has_pseudo_template else "0",
                "pseudo_template_id": example.pseudo_template_id,
                "template_supervision_source": example.template_supervision_source,
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


def run_semi_template_training(config: dict, *, create_toy_data: bool = False) -> dict:
    experiment_config = config.get("experiment", {})
    seed = int(experiment_config.get("seed", config.get("seed", 0)))
    set_seed(seed)
    output_dir = Path(experiment_config.get("output_dir", config.get("output_dir", "results/semi_template")))
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_inputs(config, create_toy_data=create_toy_data)
    assignment_report = verify_assignment_report(config["pseudo_templates"]["assignment_report"])
    if not bool(config["pseudo_templates"].get("frozen", True)):
        raise ValueError("Only frozen pseudo-template labels are supported in this training entrypoint")

    template_vocab = TemplateVocab.load(config["templates"]["template_vocab"])
    template_vocab_path = output_dir / "template_vocab.json"
    template_vocab.save(template_vocab_path)
    dataset = build_dataset(config, template_vocab)

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
        lambda_pseudo=float(loss_config.get("lambda_pseudo", 0.5)),
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
        metrics[split] = evaluate_semi_template_predictions(
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
            metrics[split]["final_loss_pseudo"] = history[-1]["loss_pseudo"]
            metrics[split]["final_loss_total"] = history[-1]["loss_total"]
            metrics[split]["train_num_gold_template_examples"] = history[-1]["num_gold_template_examples"]
            metrics[split]["train_num_pseudo_template_examples"] = history[-1]["num_pseudo_template_examples"]
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
            "assignment_report": assignment_report,
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
        "pseudo_labels_frozen": True,
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
    parser.add_argument("--config", default="configs/semi_template.yaml")
    parser.add_argument("--create-toy-data", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--lambda-gold", type=float)
    parser.add_argument("--lambda-pseudo", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.lambda_gold is not None:
        config.setdefault("loss", {})["lambda_gold"] = args.lambda_gold
    if args.lambda_pseudo is not None:
        config.setdefault("loss", {})["lambda_pseudo"] = args.lambda_pseudo
    result = run_semi_template_training(config, create_toy_data=args.create_toy_data)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
