import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechrep.models.template_head import TemplateHead


def test_template_head_outputs_batch_by_template_logits():
    head = TemplateHead(input_dim=8, num_templates=3, hidden_dim=5)
    pair_representation = torch.randn(4, 8)

    logits = head(pair_representation)

    assert tuple(logits.shape) == (4, 3)


def test_template_head_rejects_non_matrix_representation():
    head = TemplateHead(input_dim=8, num_templates=3)

    with pytest.raises(ValueError, match="batch_size x input_dim"):
        head(torch.randn(2, 2, 8))
