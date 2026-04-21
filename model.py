"""
Interpretable Seizure Prediction Model
=======================================
Multi-level interpretable EEG-based seizure prediction model.

Architecture (dual-branch + four explanation heads + risk head):

    Input  X  [B, N, C, T]
           │
    ┌──────┴──────┐
    │             │
RawEncoder    FeatEncoder
[B,N,D1]      [B,N,D2]
    │             │
    └──────┬──────┘
         Fusion
        [B, N, D]
           │
    ┌──────┼──────┬──────┐
    │      │      │      │
 FeatH  ChanH TempH ProtoH
    │      │      │      │
    └──────┴──┬───┴──────┘
           RiskHead
           [B, 1]

Tensor shape conventions:
  B – batch size
  N – number of consecutive windows
  C – EEG channels
  T – time samples per window
  D – fused model dimension
  K – handcrafted features per channel
  Kc– concept embedding dimension
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw-signal encoder  (per-channel 1-D CNN)
# ─────────────────────────────────────────────────────────────────────────────

class RawSignalEncoder(nn.Module):
    """
    Per-channel 1-D CNN that encodes each EEG channel independently.

    Each channel is processed by the same shared 1-D CNN; the per-channel
    embeddings are aggregated with a learned attention to obtain the
    window-level representation.

    Input :  x         [B, N, C, T]
    Output:  z_raw     [B, N, raw_dim]
             ch_embed  [B, N, C, ch_dim]
    """

    def __init__(self, num_channels: int, raw_dim: int, ch_dim: int) -> None:
        super().__init__()
        self.ch_dim = ch_dim
        self.raw_dim = raw_dim

        # Shared 1-D CNN applied to every channel independently
        self.channel_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, ch_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch_dim),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Attention over channels → window-level aggregation
        self.channel_attn = nn.Linear(ch_dim, 1)

        # Optional projection to raw_dim (allows raw_dim ≠ ch_dim)
        self.raw_proj = nn.Linear(ch_dim, raw_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C, T = x.shape

        # Treat every (batch, window, channel) as an independent 1-D sample
        x_flat = x.reshape(B * N * C, 1, T)            # [B*N*C, 1, T]
        h = self.channel_conv(x_flat)                   # [B*N*C, ch_dim, T]
        h_pool = self.global_pool(h).squeeze(-1)        # [B*N*C, ch_dim]

        # Per-channel embeddings
        ch_embed = h_pool.reshape(B * N, C, self.ch_dim)  # [B*N, C, ch_dim]

        # Attention-weighted sum over channels
        attn = torch.softmax(
            self.channel_attn(ch_embed).squeeze(-1), dim=-1
        ).unsqueeze(-1)                                    # [B*N, C, 1]

        z_flat = (ch_embed * attn).sum(dim=1)             # [B*N, ch_dim]
        z_raw = self.raw_proj(z_flat).reshape(B, N, -1)   # [B, N, raw_dim]
        ch_embed = ch_embed.reshape(B, N, C, self.ch_dim) # [B, N, C, ch_dim]

        return z_raw, ch_embed


# ─────────────────────────────────────────────────────────────────────────────
# 2. Handcrafted-feature / concept encoder
# ─────────────────────────────────────────────────────────────────────────────

class FeatureConceptEncoder(nn.Module):
    """
    MLP that maps handcrafted EEG features to a concept embedding.

    Channels are pooled (mean) before the MLP so the encoder is
    independent of channel count – this simplifies the architecture
    while still providing a concept-level interpretable representation.

    Input :  feat         [B, N, C, K]
    Output:  z_feat       [B, N, feat_hidden_dim]
             feat_concept [B, N, concept_dim]
    """

    def __init__(
        self, in_feat_dim: int, hidden_dim: int, concept_dim: int
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.concept_head = nn.Linear(hidden_dim, concept_dim)

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: [B, N, C, K]  →  mean over C  →  [B, N, K]
        feat_agg = feat.mean(dim=2)              # [B, N, K]
        h = self.mlp(feat_agg)                   # [B, N, hidden_dim]
        feat_concept = self.concept_head(h)      # [B, N, concept_dim]
        return h, feat_concept


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fusion layer
# ─────────────────────────────────────────────────────────────────────────────

class FusionLayer(nn.Module):
    """
    Concatenate the two branch outputs and map to a shared d_model space.

    Input :  z_raw  [B, N, d_raw]
             z_feat [B, N, d_feat]
    Output:  z      [B, N, d_model]
    """

    def __init__(self, d_raw: int, d_feat: int, d_model: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_raw + d_feat, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, z_raw: torch.Tensor, z_feat: torch.Tensor
    ) -> torch.Tensor:
        return self.fusion(torch.cat([z_raw, z_feat], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature explanation head
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExplanationHead(nn.Module):
    """
    Produces a soft importance weight over concept dimensions.

    A linear gate learns which concepts drive the current risk prediction.

    Input :  feat_concept    [B, N, Kc]
    Output:  feat_importance [B, N, Kc]  (sums to 1 over Kc)
    """

    def __init__(self, concept_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(concept_dim, concept_dim)

    def forward(self, feat_concept: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(feat_concept), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Channel explanation head
# ─────────────────────────────────────────────────────────────────────────────

class ChannelExplanationHead(nn.Module):
    """
    Scores each EEG channel by its contribution to the risk prediction.

    Input :  ch_embed      [B, N, C, ch_dim]
    Output:  ch_importance [B, N, C]  (sums to 1 over C)
    """

    def __init__(self, ch_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(ch_dim, 1)

    def forward(self, ch_embed: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(ch_embed).squeeze(-1)   # [B, N, C]
        return torch.softmax(scores, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Temporal explanation head
# ─────────────────────────────────────────────────────────────────────────────

class TemporalExplanationHead(nn.Module):
    """
    GRU over N consecutive windows + attention to identify which windows
    are most informative for the risk prediction.

    Input :  z              [B, N, D]
    Output:  temp_importance [B, N]      (sums to 1 over N)
             z_context       [B, D]      (attention-weighted GRU output)
    """

    def __init__(self, d_model: int, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.gru(z)                               # [B, N, D]
        scores = self.attn(h).squeeze(-1)                # [B, N]
        temp_importance = torch.softmax(scores, dim=-1)  # [B, N]
        z_context = (h * temp_importance.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return temp_importance, z_context


# ─────────────────────────────────────────────────────────────────────────────
# 7. Prototype (case) explanation head
# ─────────────────────────────────────────────────────────────────────────────

class PrototypeExplanationHead(nn.Module):
    """
    Maintains a learnable memory bank of P prototype embeddings.
    Retrieves the Top-k most similar prototypes for each query.

    Input :  z_context  [B, D]
    Output:  proto_sim  [B, P]   (cosine similarity to every prototype)
             topk_idx   [B, k]   (indices of top-k prototypes)
             topk_val   [B, k]   (similarity scores of top-k prototypes)
    """

    def __init__(self, d_model: int, num_prototypes: int) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        # Learnable prototypes – initialised with small noise
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, d_model) * 0.02
        )
        # Integer label for each prototype (0=interictal, 1=preictal).
        # Stored as a non-trained buffer so it moves with the model.
        self.register_buffer(
            "prototype_labels",
            torch.zeros(num_prototypes, dtype=torch.long),
        )

    def forward(
        self, z_context: torch.Tensor, topk: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_norm = F.normalize(z_context, dim=-1)           # [B, D]
        p_norm = F.normalize(self.prototypes, dim=-1)     # [P, D]
        sim = torch.matmul(z_norm, p_norm.t())            # [B, P]
        k = min(topk, self.num_prototypes)
        topk_val, topk_idx = torch.topk(sim, k=k, dim=-1)
        return sim, topk_idx, topk_val

    def update_prototype_labels(self, labels: torch.Tensor) -> None:
        """Set prototype class labels (called after k-means initialisation)."""
        self.prototype_labels.copy_(labels)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Risk head
# ─────────────────────────────────────────────────────────────────────────────

class RiskHead(nn.Module):
    """
    Combines the temporal context vector and the importance-weighted concept
    summary to produce a scalar seizure risk logit.

    Input :  z_context    [B, D]
             feat_summary [B, Kc]
    Output:  risk_logit   [B, 1]
    """

    def __init__(self, d_model: int, concept_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model + concept_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, z_context: torch.Tensor, feat_summary: torch.Tensor
    ) -> torch.Tensor:
        return self.fc(torch.cat([z_context, feat_summary], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full model
# ─────────────────────────────────────────────────────────────────────────────

class InterpretableSeizurePredictor(nn.Module):
    """
    Multi-level interpretable model for EEG-based seizure prediction.

    The model outputs:
      - ``risk_logit`` / ``risk_prob``    – seizure risk score
      - ``feat_importance``               – concept-feature importance [B,N,Kc]
      - ``channel_importance``            – EEG channel importance      [B,N,C]
      - ``temporal_importance``           – window-level importance      [B,N]
      - ``prototype_similarity``          – similarity to all prototypes [B,P]
      - ``prototype_topk_idx/val``        – top-k prototype retrieval

    Args:
        num_channels:    Number of EEG channels (C)
        raw_dim:         Raw encoder output dimension (D1)
        ch_dim:          Per-channel embedding dimension
        feat_dim:        Handcrafted features per channel (K)
        feat_hidden_dim: Feature-encoder hidden dimension (D2)
        concept_dim:     Concept embedding dimension (Kc)
        d_model:         Fused model dimension (D)
        num_prototypes:  Prototype memory bank size (P)
    """

    def __init__(
        self,
        num_channels: int = 19,
        raw_dim: int = 128,
        ch_dim: int = 64,
        feat_dim: int = 15,
        feat_hidden_dim: int = 128,
        concept_dim: int = 32,
        d_model: int = 256,
        num_prototypes: int = 50,
    ) -> None:
        super().__init__()

        self.raw_encoder = RawSignalEncoder(
            num_channels=num_channels,
            raw_dim=raw_dim,
            ch_dim=ch_dim,
        )
        self.feat_encoder = FeatureConceptEncoder(
            in_feat_dim=feat_dim,
            hidden_dim=feat_hidden_dim,
            concept_dim=concept_dim,
        )
        self.fusion = FusionLayer(
            d_raw=raw_dim,
            d_feat=feat_hidden_dim,
            d_model=d_model,
        )
        self.feature_head = FeatureExplanationHead(concept_dim=concept_dim)
        self.channel_head = ChannelExplanationHead(ch_dim=ch_dim)
        self.temporal_head = TemporalExplanationHead(d_model=d_model)
        self.prototype_head = PrototypeExplanationHead(
            d_model=d_model,
            num_prototypes=num_prototypes,
        )
        self.risk_head = RiskHead(d_model=d_model, concept_dim=concept_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        feat: torch.Tensor,
        topk_proto: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:          Raw EEG window sequences  [B, N, C, T]
            feat:       Handcrafted features       [B, N, C, K]
            topk_proto: Number of top prototypes to retrieve

        Returns:
            Dictionary of tensors (predictions + explanations).
        """
        # ── Branch 1: raw-signal encoding ──────────────────────────────
        z_raw, ch_embed = self.raw_encoder(x)           # [B,N,D1], [B,N,C,D_ch]

        # ── Branch 2: handcrafted-feature encoding ──────────────────────
        z_feat, feat_concept = self.feat_encoder(feat)  # [B,N,D2], [B,N,Kc]

        # ── Fusion ──────────────────────────────────────────────────────
        z = self.fusion(z_raw, z_feat)                  # [B, N, D]

        # ── Explanation heads ────────────────────────────────────────────
        feat_importance = self.feature_head(feat_concept)   # [B, N, Kc]
        ch_importance   = self.channel_head(ch_embed)       # [B, N, C]
        temp_importance, z_context = self.temporal_head(z)  # [B,N], [B,D]

        proto_sim, topk_idx, topk_val = self.prototype_head(
            z_context, topk=topk_proto
        )

        # Importance-weighted concept summary (collapsed over time)
        feat_summary = (feat_concept * feat_importance).sum(dim=1)  # [B, Kc]

        # ── Risk prediction ──────────────────────────────────────────────
        risk_logit = self.risk_head(z_context, feat_summary)  # [B, 1]
        risk_prob  = torch.sigmoid(risk_logit)

        return {
            "risk_logit":          risk_logit,
            "risk_prob":           risk_prob,
            "feat_importance":     feat_importance,
            "channel_importance":  ch_importance,
            "temporal_importance": temp_importance,
            "prototype_similarity": proto_sim,
            "prototype_topk_idx":  topk_idx,
            "prototype_topk_val":  topk_val,
            "z_context":           z_context,
        }

    # ------------------------------------------------------------------

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
