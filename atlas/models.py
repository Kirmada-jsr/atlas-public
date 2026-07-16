"""Atlas model definitions.

Single source of truth for the three trainable modules. Architectures are
checkpoint-compatible with the released Atlas weights: do not change layer
shapes or module names, or ``load_state_dict`` will fail.

All modules operate directly on raw SONAR embeddings (dim 1024). There is no
intermediate latent space.

v0.2.0 replaces the single K-to-1 composer with two heads. The grouping
model decides which retrieved facts can share one fluent output sentence;
the fusion model folds each group of up to 3 facts into one SONAR vector,
decoded to one sentence per group.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SONAR_DIM = 1024
QE_HIDDEN = 1024
GROUPER_HIDDEN = 256
FUSER_HIDDEN = 1024
FUSER_LAYERS = 4
FUSER_HEADS = 8


class QueryEncoder(nn.Module):
    """Maps a question SONAR embedding to a query vector in SONAR space.

    LayerNorm -> Linear -> GELU -> Linear -> GELU -> Linear, then L2
    normalization so the query lives on the unit sphere alongside the
    (normalized) fact-pool vectors.
    """

    def __init__(self, sonar_dim: int = SONAR_DIM, hidden: int = QE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(sonar_dim),
            nn.Linear(sonar_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, sonar_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class GroupingModel(nn.Module):
    """Pairwise fusability classifier over two fact embeddings.

    Answers "can these two distinct facts share one fluent sentence?", which
    is a different question from "are they near-duplicates" (near-duplicate
    removal is done by a plain cosine threshold in the pipeline). A shared
    projection ``phi`` embeds each fact; the classifier scores the pair from
    ``[phi(a)+phi(b), phi(a)*phi(b), |phi(a)-phi(b)|]``, a combination that
    makes the score symmetric in (a, b). Output is one logit; the pipeline
    applies a sigmoid and thresholds it.
    """

    def __init__(self, sonar_dim: int = SONAR_DIM, hidden: int = GROUPER_HIDDEN):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(sonar_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.cls = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pa, pb = self.phi(a), self.phi(b)
        pair = torch.cat([pa + pb, pa * pb, (pa - pb).abs()], dim=-1)
        return self.cls(pair).squeeze(-1)


class FusionModel(nn.Module):
    """Fuses a group of up to 3 fact SONAR embeddings into one SONAR vector.

    Input projection, pre-LayerNorm TransformerEncoder over the group
    members (padding slots hidden via the key-padding mask), mask-aware mean
    pooling, then an output MLP back to SONAR space. The output vector is
    decodable by the SONAR decoder into one fluent sentence covering the
    group.
    """

    def __init__(
        self,
        sonar_dim: int = SONAR_DIM,
        hidden: int = FUSER_HIDDEN,
        n_layers: int = FUSER_LAYERS,
        n_heads: int = FUSER_HEADS,
    ):
        super().__init__()
        self.inp = nn.Linear(sonar_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=4 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False only silences a construction-time warning:
        # torch disables the nested-tensor fast path for norm_first layers
        # anyway, so the compute path is identical.
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, sonar_dim),
        )

    def forward(
        self,
        x: torch.Tensor,      # [B, N, D] group members, zero-padded to N slots
        mask: torch.Tensor,   # [B, N] True = real member, False = padding
    ) -> torch.Tensor:        # [B, D]
        h = self.tf(self.inp(x), src_key_padding_mask=~mask)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.out(pooled)
