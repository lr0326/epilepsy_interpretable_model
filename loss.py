"""
Multi-task loss function for interpretable seizure prediction.

Total loss = pred_loss
           + λ_feat  * feat_entropy
           + λ_ch    * ch_entropy
           + λ_temp  * temp_entropy
           + λ_proto * proto_conf

Regularisation terms encourage:
  feat_entropy  – sparse feature attention (few concepts dominate)
  ch_entropy    – sparse channel attention (few channels dominate)
  temp_entropy  – focused temporal attention (few windows dominate)
  proto_conf    – high similarity to at least one prototype
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_feat: float = 1e-3,
    lambda_ch: float = 1e-3,
    lambda_temp: float = 1e-3,
    lambda_proto: float = 1e-3,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the multi-task training loss.

    Args:
        outputs:      Dict returned by ``InterpretableSeizurePredictor.forward``.
        labels:       Binary ground-truth labels  [B, 1]  float.
        lambda_feat:  Weight for feature-attention entropy regularisation.
        lambda_ch:    Weight for channel-attention entropy regularisation.
        lambda_temp:  Weight for temporal-attention entropy regularisation.
        lambda_proto: Weight for prototype-confidence regularisation.
        pos_weight:   Optional positive-class weight tensor for BCE loss
                      (useful when the dataset is imbalanced).

    Returns:
        Dict with keys ``total_loss``, ``pred_loss``, ``feat_entropy``,
        ``ch_entropy``, ``temp_entropy``, ``proto_conf``.
    """
    risk_logit      = outputs["risk_logit"]            # [B, 1]
    feat_importance = outputs["feat_importance"]       # [B, N, Kc]
    ch_importance   = outputs["channel_importance"]    # [B, N, C]
    temp_importance = outputs["temporal_importance"]   # [B, N]
    proto_sim       = outputs["prototype_similarity"]  # [B, P]

    # ── 1. Main prediction loss ─────────────────────────────────────────
    pred_loss = F.binary_cross_entropy_with_logits(
        risk_logit, labels.float(), pos_weight=pos_weight
    )

    # ── 2. Feature-attention entropy (encourages sparse concepts) ───────
    feat_entropy = -(
        feat_importance * (feat_importance + 1e-8).log()
    ).sum(dim=-1).mean()

    # ── 3. Channel-attention entropy (encourages focal channel weights) ──
    ch_entropy = -(
        ch_importance * (ch_importance + 1e-8).log()
    ).sum(dim=-1).mean()

    # ── 4. Temporal-attention entropy (encourages focused windows) ───────
    temp_entropy = -(
        temp_importance * (temp_importance + 1e-8).log()
    ).sum(dim=-1).mean()

    # ── 5. Prototype confidence (rewards high max similarity) ───────────
    proto_conf = -proto_sim.max(dim=-1).values.mean()

    total_loss = (
        pred_loss
        + lambda_feat  * feat_entropy
        + lambda_ch    * ch_entropy
        + lambda_temp  * temp_entropy
        + lambda_proto * proto_conf
    )

    return {
        "total_loss":  total_loss,
        "pred_loss":   pred_loss,
        "feat_entropy": feat_entropy,
        "ch_entropy":  ch_entropy,
        "temp_entropy": temp_entropy,
        "proto_conf":  proto_conf,
    }
