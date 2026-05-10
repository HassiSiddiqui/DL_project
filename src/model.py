"""
src/model.py
------------
DTFD-MIL model components used in both Assignment 2 (baseline) and
Assignment 3 (proposed Semantic-DTFD-MIL).

Components:
  - DimReduction        : linear projection into hidden space
  - Attention_Gated     : gated self-attention (Tier-1)
  - Attention_with_Classifier : Tier-2 global bag classifier
  - Classifier_1fc      : simple 1-FC classification head
  - SemanticDTFDMIL     : full proposed pipeline wrapper
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Re-export from project Model/ for convenience ────────────────────────────
from Model.network import Classifier_1fc, DimReduction
from Model.Attention import Attention_Gated, Attention_with_Classifier


class SemanticDTFDMIL(nn.Module):
    """
    Proposed full pipeline (Assignment 3):
      Input : bag of N patches, each a 512-dim PLIP feature vector
      Output: class logits (2,) — Normal vs Tumor

    Architecture (Double-Tier):
      Tier 1 — per pseudo-bag local attention + distillation
      Tier 2 — global attention across distilled pseudo-bag reps + classify
    """

    def __init__(
        self,
        in_chn: int   = 512,
        m_dim: int    = 256,
        num_cls: int  = 2,
        att_dim: int  = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_reduction = DimReduction(in_chn, m_dim, numLayer_Res=0)
        self.attention      = Attention_Gated(L=m_dim, D=att_dim, K=1)
        self.att_classifier = Attention_with_Classifier(
            L=m_dim, D=att_dim, K=1, num_cls=num_cls, droprate=dropout
        )
        self.classifier     = Classifier_1fc(m_dim, num_cls, droprate=dropout)

    def forward_bag(self, pseudo_bag: torch.Tensor):
        """Process a single pseudo-bag → (1, m_dim) distilled representation."""
        feat = self.dim_reduction(pseudo_bag)   # (N, m_dim)
        att  = self.attention(feat)             # (1, N)
        dist = torch.mm(att, feat)              # (1, m_dim)
        return dist, self.classifier(dist)      # dist, tier1_logits

    def forward(self, pseudo_bags: list[torch.Tensor]):
        """
        Args:
            pseudo_bags : list of K tensors, each (Nk, in_chn)
        Returns:
            tier2_logits : (K, num_cls)  — one per pseudo-bag (final prediction)
            tier1_logits : (K, num_cls)  — per-bag head (auxiliary loss)
            distilled    : (K, m_dim)   — distilled representations
        """
        dists, t1_logits = [], []
        for pb in pseudo_bags:
            d, l = self.forward_bag(pb)
            dists.append(d)
            t1_logits.append(l)

        distilled    = torch.cat(dists, dim=0)          # (K, m_dim)
        tier2_logits = self.att_classifier(distilled)   # (K, num_cls)
        tier1_logits = torch.cat(t1_logits, dim=0)

        return tier2_logits, tier1_logits, distilled


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torchinfo import summary
    model = SemanticDTFDMIL()
    print(f"SemanticDTFDMIL — trainable params: {count_parameters(model):,}")
    # simulate K=5 pseudo-bags each with 4 patches
    bags = [torch.randn(4, 512) for _ in range(5)]
    t2, t1, dist = model(bags)
    print(f"Tier-2 logits : {t2.shape}")
    print(f"Tier-1 logits : {t1.shape}")
    print(f"Distilled     : {dist.shape}")
