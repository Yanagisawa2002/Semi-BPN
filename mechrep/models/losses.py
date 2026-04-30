"""Loss functions for supervised template regularization."""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise RuntimeError("Loss functions require PyTorch") from exc

from mechrep.data.gold_template_dataset import IGNORE_TEMPLATE_INDEX


def pair_prediction_loss(pair_score: torch.Tensor, pair_label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pair_score.float(), pair_label.float())


def gold_template_loss(
    template_logits: torch.Tensor,
    primary_template_index: torch.Tensor,
    pair_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = (primary_template_index != IGNORE_TEMPLATE_INDEX) & (pair_label.long() == 1)
    if int(valid_mask.sum().item()) == 0:
        return template_logits.sum() * 0.0, valid_mask
    return F.cross_entropy(template_logits[valid_mask], primary_template_index[valid_mask].long()), valid_mask


def joint_gold_template_loss(
    pair_score: torch.Tensor,
    pair_label: torch.Tensor,
    template_logits: torch.Tensor,
    primary_template_index: torch.Tensor,
    *,
    lambda_gold: float,
) -> dict:
    if lambda_gold < 0:
        raise ValueError(f"lambda_gold must be non-negative, got {lambda_gold}")
    loss_pred = pair_prediction_loss(pair_score, pair_label)
    raw_loss_gold, valid_mask = gold_template_loss(template_logits, primary_template_index, pair_label)
    if lambda_gold == 0:
        loss_gold = raw_loss_gold * 0.0
    else:
        loss_gold = raw_loss_gold
    loss_total = loss_pred + float(lambda_gold) * loss_gold
    return {
        "loss_pred": loss_pred,
        "loss_gold": loss_gold,
        "loss_total": loss_total,
        "num_gold_template_examples_in_batch": int(valid_mask.sum().item()),
        "gold_template_mask": valid_mask,
    }


def _masked_template_ce(
    template_logits: torch.Tensor,
    template_index: torch.Tensor,
    pair_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = (template_index != IGNORE_TEMPLATE_INDEX) & (pair_label.long() == 1)
    if int(valid_mask.sum().item()) == 0:
        return template_logits.sum() * 0.0, valid_mask
    return F.cross_entropy(template_logits[valid_mask], template_index[valid_mask].long()), valid_mask


def compute_semi_template_loss(
    pair_score: torch.Tensor,
    pair_label: torch.Tensor,
    template_logits: torch.Tensor,
    gold_template_index: torch.Tensor,
    pseudo_template_index: torch.Tensor,
    *,
    lambda_gold: float,
    lambda_pseudo: float,
) -> dict:
    if lambda_gold < 0:
        raise ValueError(f"lambda_gold must be non-negative, got {lambda_gold}")
    if lambda_pseudo < 0:
        raise ValueError(f"lambda_pseudo must be non-negative, got {lambda_pseudo}")

    loss_pred = pair_prediction_loss(pair_score, pair_label)
    raw_loss_gold, gold_mask = _masked_template_ce(template_logits, gold_template_index, pair_label)
    raw_loss_pseudo, pseudo_mask = _masked_template_ce(template_logits, pseudo_template_index, pair_label)
    loss_gold = raw_loss_gold * 0.0 if lambda_gold == 0 else raw_loss_gold
    loss_pseudo = raw_loss_pseudo * 0.0 if lambda_pseudo == 0 else raw_loss_pseudo
    loss_total = loss_pred + float(lambda_gold) * loss_gold + float(lambda_pseudo) * loss_pseudo
    return {
        "loss_total": loss_total,
        "loss_pred": loss_pred,
        "loss_gold": loss_gold,
        "loss_pseudo": loss_pseudo,
        "num_gold_template_examples": int(gold_mask.sum().item()),
        "num_pseudo_template_examples": int(pseudo_mask.sum().item()),
        "gold_template_mask": gold_mask,
        "pseudo_template_mask": pseudo_mask,
    }
