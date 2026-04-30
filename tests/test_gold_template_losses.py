import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.data.gold_template_dataset import IGNORE_TEMPLATE_INDEX
from mechrep.models.losses import gold_template_loss, joint_gold_template_loss, pair_prediction_loss


def test_gold_template_loss_masks_missing_and_negative_examples():
    template_logits = torch.tensor(
        [
            [4.0, 0.1],
            [0.2, 3.0],
            [0.3, 2.0],
        ]
    )
    primary_template_index = torch.tensor([0, 1, IGNORE_TEMPLATE_INDEX])
    pair_label = torch.tensor([1.0, 0.0, 1.0])

    loss, mask = gold_template_loss(template_logits, primary_template_index, pair_label)

    expected = torch.nn.functional.cross_entropy(template_logits[:1], torch.tensor([0]))
    assert torch.allclose(loss, expected)
    assert mask.tolist() == [True, False, False]


def test_lambda_gold_zero_reproduces_prediction_only_loss():
    pair_score = torch.tensor([0.1, -0.2, 0.3])
    pair_label = torch.tensor([1.0, 0.0, 1.0])
    template_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.3, 0.7]])
    primary_template_index = torch.tensor([0, 1, 1])

    result = joint_gold_template_loss(
        pair_score,
        pair_label,
        template_logits,
        primary_template_index,
        lambda_gold=0.0,
    )

    assert torch.allclose(result["loss_total"], pair_prediction_loss(pair_score, pair_label))
    assert torch.allclose(result["loss_gold"], torch.tensor(0.0))


def test_batches_with_no_gold_template_examples_do_not_crash():
    pair_score = torch.tensor([0.1, -0.2])
    pair_label = torch.tensor([1.0, 0.0])
    template_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    primary_template_index = torch.tensor([IGNORE_TEMPLATE_INDEX, IGNORE_TEMPLATE_INDEX])

    result = joint_gold_template_loss(
        pair_score,
        pair_label,
        template_logits,
        primary_template_index,
        lambda_gold=1.0,
    )

    assert result["num_gold_template_examples_in_batch"] == 0
    assert torch.allclose(result["loss_gold"], torch.tensor(0.0))
    assert torch.isfinite(result["loss_total"])
