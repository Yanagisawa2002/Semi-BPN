import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.gold_template_dataset import IGNORE_TEMPLATE_INDEX
from mechrep.models.losses import compute_semi_template_loss, pair_prediction_loss


def test_semi_loss_masks_gold_and_pseudo_examples_separately():
    pair_score = torch.tensor([1.0, 0.5, -0.5, 0.2])
    pair_label = torch.tensor([1.0, 1.0, 1.0, 0.0])
    template_logits = torch.tensor(
        [
            [3.0, 0.1],
            [0.2, 2.5],
            [0.5, 0.6],
            [0.1, 3.0],
        ]
    )
    gold_index = torch.tensor([0, IGNORE_TEMPLATE_INDEX, IGNORE_TEMPLATE_INDEX, IGNORE_TEMPLATE_INDEX])
    pseudo_index = torch.tensor([IGNORE_TEMPLATE_INDEX, 1, IGNORE_TEMPLATE_INDEX, 1])

    result = compute_semi_template_loss(
        pair_score,
        pair_label,
        template_logits,
        gold_index,
        pseudo_index,
        lambda_gold=1.0,
        lambda_pseudo=0.5,
    )

    assert result["num_gold_template_examples"] == 1
    assert result["num_pseudo_template_examples"] == 1
    assert torch.allclose(result["loss_gold"], torch.nn.functional.cross_entropy(template_logits[:1], torch.tensor([0])))
    assert torch.allclose(result["loss_pseudo"], torch.nn.functional.cross_entropy(template_logits[1:2], torch.tensor([1])))


def test_semi_loss_no_gold_or_no_pseudo_batches_do_not_crash():
    pair_score = torch.tensor([0.1, -0.2])
    pair_label = torch.tensor([1.0, 0.0])
    template_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ignore = torch.tensor([IGNORE_TEMPLATE_INDEX, IGNORE_TEMPLATE_INDEX])

    result = compute_semi_template_loss(
        pair_score,
        pair_label,
        template_logits,
        ignore,
        ignore,
        lambda_gold=1.0,
        lambda_pseudo=1.0,
    )

    assert result["num_gold_template_examples"] == 0
    assert result["num_pseudo_template_examples"] == 0
    assert torch.allclose(result["loss_gold"], torch.tensor(0.0))
    assert torch.allclose(result["loss_pseudo"], torch.tensor(0.0))
    assert torch.isfinite(result["loss_total"])


def test_lambda_zero_switches_disable_template_losses():
    pair_score = torch.tensor([0.1, 0.2, -0.3])
    pair_label = torch.tensor([1.0, 1.0, 0.0])
    template_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.1]])
    gold_index = torch.tensor([0, IGNORE_TEMPLATE_INDEX, IGNORE_TEMPLATE_INDEX])
    pseudo_index = torch.tensor([IGNORE_TEMPLATE_INDEX, 1, IGNORE_TEMPLATE_INDEX])

    no_pseudo = compute_semi_template_loss(
        pair_score,
        pair_label,
        template_logits,
        gold_index,
        pseudo_index,
        lambda_gold=1.0,
        lambda_pseudo=0.0,
    )
    pred_only = compute_semi_template_loss(
        pair_score,
        pair_label,
        template_logits,
        gold_index,
        pseudo_index,
        lambda_gold=0.0,
        lambda_pseudo=0.0,
    )

    assert torch.allclose(no_pseudo["loss_pseudo"], torch.tensor(0.0))
    assert torch.allclose(pred_only["loss_gold"], torch.tensor(0.0))
    assert torch.allclose(pred_only["loss_pseudo"], torch.tensor(0.0))
    assert torch.allclose(pred_only["loss_total"], pair_prediction_loss(pair_score, pair_label))
