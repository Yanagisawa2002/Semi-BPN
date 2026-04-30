import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.training.train_semi_supervised import load_config, run_semi_template_training


def _workspace_tmp_dir():
    path = Path(__file__).resolve().parents[1] / ".test_tmp" / f"semi_train_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_semi_supervised_training_runs_on_toy_dataset_and_writes_outputs():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "semi_template.yaml")
        config["experiment"]["output_dir"] = str(tmp_dir / "out")
        config["training"]["epochs"] = 3
        config["training"]["batch_size"] = 2
        config["model"]["embedding_dim"] = 8
        config["model"]["hidden_dim"] = 16
        result = run_semi_template_training(config, create_toy_data=True)

        out_dir = Path(result["output_dir"])
        assert (out_dir / "predictions_train.tsv").exists()
        assert (out_dir / "predictions_valid.tsv").exists()
        assert (out_dir / "predictions_test.tsv").exists()
        assert (out_dir / "metrics_train.json").exists()
        assert (out_dir / "metrics_valid.json").exists()
        assert (out_dir / "metrics_test.json").exists()
        assert (out_dir / "config.yaml").exists()
        assert (out_dir / "template_vocab.json").exists()

        metrics = json.loads((out_dir / "metrics_train.json").read_text(encoding="utf-8"))
        assert "final_loss_pred" in metrics
        assert "final_loss_gold" in metrics
        assert "final_loss_pseudo" in metrics
        assert "final_loss_total" in metrics
        assert metrics["train_num_gold_template_examples"] > 0
        assert metrics["train_num_pseudo_template_examples"] > 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_semi_supervised_sanity_lambda_zero_modes_run():
    tmp_dir = _workspace_tmp_dir()
    try:
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "semi_template.yaml")
        config["experiment"]["output_dir"] = str(tmp_dir / "out")
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 2
        config["model"]["embedding_dim"] = 8
        config["model"]["hidden_dim"] = 16

        gold_only = dict(config)
        gold_only["loss"] = dict(config["loss"])
        gold_only["loss"]["lambda_pseudo"] = 0.0
        result_gold_only = run_semi_template_training(gold_only, create_toy_data=True)
        metrics_gold_only = json.loads((Path(result_gold_only["output_dir"]) / "metrics_train.json").read_text(encoding="utf-8"))
        assert metrics_gold_only["final_loss_pseudo"] == 0.0

        pred_only = load_config(Path(__file__).resolve().parents[1] / "configs" / "semi_template.yaml")
        pred_only["experiment"]["output_dir"] = str(tmp_dir / "out_pred")
        pred_only["training"]["epochs"] = 2
        pred_only["training"]["batch_size"] = 2
        pred_only["model"]["embedding_dim"] = 8
        pred_only["model"]["hidden_dim"] = 16
        pred_only["loss"]["lambda_gold"] = 0.0
        pred_only["loss"]["lambda_pseudo"] = 0.0
        result_pred_only = run_semi_template_training(pred_only, create_toy_data=True)
        metrics_pred_only = json.loads((Path(result_pred_only["output_dir"]) / "metrics_train.json").read_text(encoding="utf-8"))
        assert metrics_pred_only["final_loss_gold"] == 0.0
        assert metrics_pred_only["final_loss_pseudo"] == 0.0
        assert metrics_pred_only["final_loss_total"] == metrics_pred_only["final_loss_pred"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
