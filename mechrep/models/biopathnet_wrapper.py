"""Safe pair-scoring wrapper used by the endpoint-level BioPathNet adapters."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise RuntimeError("BioPathNet wrappers require PyTorch") from exc


class PairEmbeddingBioPathNetWrapper(nn.Module):
    """Endpoint pair encoder exposing pair scores and pair representations.

    The original BioPathNet pipeline is kept untouched. This wrapper is the
    currently runnable pair-level backend in this repository: it consumes the
    adapted drug and endpoint IDs, embeds them, and exposes a representation
    that downstream supervised heads can use.
    """

    def __init__(self, num_drugs: int, num_endpoints: int, embedding_dim: int = 64):
        super().__init__()
        if num_drugs <= 0 or num_endpoints <= 0:
            raise ValueError("num_drugs and num_endpoints must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.drug_embedding = nn.Embedding(num_drugs, embedding_dim)
        self.endpoint_embedding = nn.Embedding(num_endpoints, embedding_dim)
        self.representation_dim = embedding_dim * 4
        self.pair_scorer = nn.Linear(self.representation_dim, 1)

    def pair_representation(self, drug_index: torch.Tensor, endpoint_index: torch.Tensor) -> torch.Tensor:
        drug = self.drug_embedding(drug_index)
        endpoint = self.endpoint_embedding(endpoint_index)
        return torch.cat([drug, endpoint, drug * endpoint, torch.abs(drug - endpoint)], dim=-1)

    def forward(self, drug_index: torch.Tensor, endpoint_index: torch.Tensor) -> dict:
        representation = self.pair_representation(drug_index, endpoint_index)
        pair_score = self.pair_scorer(representation).squeeze(-1)
        return {
            "pair_score": pair_score,
            "pair_representation": representation,
        }
