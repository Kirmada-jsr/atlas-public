"""Atlas model definitions.

Single source of truth for the three trainable modules. Architectures are
checkpoint-compatible with the released Atlas weights — do not change layer
shapes or module names, or ``load_state_dict`` will fail.

All modules operate directly on raw SONAR embeddings (dim 1024). There is no
intermediate latent space.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SONAR_DIM = 1024
QE_HIDDEN = 1024

# Initial value of DotProductScorer.logit_scale: log(1 / 0.07), the standard
# CLIP-style contrastive temperature. Only used during training; kept so the
# released checkpoints load cleanly.
_LOGIT_SCALE_INIT = 2.6593


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


class DotProductScorer(nn.Module):
    """Cosine-similarity scorer with a learned temperature.

    The temperature (``logit_scale``) is a training-time artifact of the
    InfoNCE objective; at inference ``pairwise`` returns plain cosine
    similarities. The parameter is retained for checkpoint compatibility.
    """

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * _LOGIT_SCALE_INIT)

    def pairwise(self, z_pool: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between every query and every pool vector.

        Args:
            z_pool:  ``[P, D]`` candidate fact embeddings.
            y_batch: ``[B, D]`` query vectors.

        Returns:
            ``[B, P]`` similarity scores (higher = more relevant).
        """
        z_n = F.normalize(z_pool, dim=-1)
        y_n = F.normalize(y_batch, dim=-1)
        return y_n @ z_n.T


class Composer(nn.Module):
    """Folds K sentence SONAR embeddings into one paragraph-level embedding.

    TransformerEncoder over the K inputs (with additive score conditioning),
    mask-aware mean pooling, then an output MLP. Output is a raw SONAR-space
    vector (no normalization) suitable for the SONAR decoder.
    """

    def __init__(
        self,
        n_heads: int = 8,
        n_layers: int = 4,
        sonar_dim: int = SONAR_DIM,
    ):
        super().__init__()
        D = sonar_dim

        # Score conditioning: scalar relevance score -> D, added to each input.
        self.score_embed = nn.Sequential(
            nn.Linear(1, D // 4), nn.GELU(),
            nn.Linear(D // 4, D),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=n_heads,
            dim_feedforward=4 * D,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mlp = nn.Sequential(
            nn.Linear(D, 2 * D), nn.GELU(),
            nn.Linear(2 * D, D),
        )

    def forward(
        self,
        z_topk: torch.Tensor,                                # [B, K, D]
        scores: torch.Tensor,                                # [B, K]
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B, K] True = pad
    ) -> torch.Tensor:                                       # [B, D]
        x = z_topk + self.score_embed(scores.unsqueeze(-1))
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.mlp(h)
