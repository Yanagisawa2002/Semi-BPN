from pathlib import Path

import yaml


def test_k50_fallback_experiment_matrix_has_runnable_variants():
    matrix_path = Path("configs/biopathnet_k50_fallback_experiment_matrix.yaml")
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))

    variants = matrix["variants"]
    assert matrix["stage_order"]
    assert set(matrix["stage_order"]) <= set(variants)
    assert "batch_d32_l2_neg4_b4" in variants
    assert "throughput_d32_l4_neg16_b4" in variants
    assert variants["batch_d32_l2_neg4_b4"]["train_batch_size"] == 4

    required_positive_ints = ("hidden_dim", "hidden_layers", "num_negative", "train_batch_size")
    seen_run_names = set()
    for name, spec in variants.items():
        for key in required_positive_ints:
            assert isinstance(spec[key], int), (name, key)
            assert spec[key] > 0, (name, key)
        assert spec["learning_rate"] > 0
        assert spec["run_name"]
        assert spec["run_name"] not in seen_run_names
        seen_run_names.add(spec["run_name"])

    assert "run_biopathnet_k50_fallback_variant.sh" in matrix["commands"]["train_and_eval_one_variant"]
