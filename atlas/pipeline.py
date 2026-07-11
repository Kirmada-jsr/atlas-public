"""Atlas inference pipeline.

``Atlas.from_pretrained()`` downloads the released weights and fact index
from the Hugging Face Hub (pinned to the revision matching this package
version) and assembles the full ask pipeline:

    question --SONAR encode--> query embedding
             --QueryEncoder + chunked cosine scan--> top-K fact sentences
             --Composer--> paragraph-level SONAR embedding
             --SONAR decode--> answer text
             (+ nearest stored passages to the composed vector, for grounding)

The retrieval / composition math is ported behavior-for-behavior from the
validated dev inference script. Public-facing outputs are identical to it.

v0.1.1: retrieval scores never enter the composer. They are used only to
SELECT which retrieved sentences are composed, via the relative drop-off
rule ``alpha``: keep sentence i iff score_i >= alpha * score_1 (top-1 is
always kept; alpha <= 0 keeps all k). The composer takes a variable number
of inputs (trained on K = 1..4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .models import Composer, DotProductScorer, QueryEncoder
from .sonar import DEFAULT_DECODER, DEFAULT_ENCODER, Sonar

DEFAULT_MODEL_REPO = "kirmada-jsr/atlas"
DEFAULT_INDEX_REPO = "kirmada-jsr/atlas-index"

# Filenames inside the HF repos (written by scripts/upload_to_hf.py).
RETRIEVER_FILE = "retriever.pt"
COMPOSER_FILE = "composer.pt"
ENCODED_FILE = "msmarco_encoded.pt"
PARA_TARGETS_FILE = "para_targets.pt"
MANIFEST_FILE = "manifest.json"

# Retained for backward compatibility only; score_mode is deprecated and
# ignored (the c0.1.1 composer takes no scores).
_SCORE_MODES = ("uniform", "retriever")


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _default_revision() -> str:
    """HF revision pinned to this package version (tag ``v<version>``)."""
    from . import __version__

    return f"v{__version__}"


@dataclass
class AtlasResult:
    """Everything one ``ask()`` call produced."""

    question: str
    answer: Optional[str]                      # decoded paragraph; None if decode=False
    retrieved: List[Tuple[str, float]]         # (sentence, raw retriever cosine score)
    retrieved_indices: List[int] = field(default_factory=list)  # pool indices of `retrieved`
    kept: List[bool] = field(default_factory=list)  # alpha-selection survivors (composed subset)
    passages: List[Dict] = field(default_factory=list)
    # each: {"question": str, "sentences": List[str], "similarity": float,
    #        "index": int}
    embedding: Optional[torch.Tensor] = None   # composed SONAR vector [D], CPU


class Atlas:
    """Pretrained Atlas: retrieval + composition over a SONAR fact memory."""

    def __init__(
        self,
        query_enc: QueryEncoder,
        scorer: DotProductScorer,
        composer: Composer,
        data: Dict,
        sonar: Sonar,
        device: torch.device,
        chunk_size: int = 4096,
        alpha: float = 0.0,
    ):
        self.query_enc = query_enc
        self.scorer = scorer
        self.composer = composer
        self.sonar = sonar
        self.device = device
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.manifest: Optional[Dict] = None

        self.questions: List[str] = data["questions"]
        self.sentences: List[str] = data["sentences"]
        self.sent_embs: torch.Tensor = data["sent_embs"]        # [S, D]
        self.sent_ranges: List[Tuple[int, int]] = data["sent_ranges"]
        self.para_targets: torch.Tensor = data["para_targets"]  # [N, D]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_repo: str = DEFAULT_MODEL_REPO,
        index_repo: str = DEFAULT_INDEX_REPO,
        revision: Optional[str] = None,
        device: str | torch.device = "auto",
        pool_device: Optional[str] = None,
        encoder_model: str = DEFAULT_ENCODER,
        decoder_model: str = DEFAULT_DECODER,
        chunk_size: int = 4096,
        alpha: float = 0.0,
    ) -> "Atlas":
        """Download weights + fact index from the HF Hub and build the pipeline.

        Args:
            model_repo:  HF model repo with ``retriever.pt`` / ``composer.pt``.
            index_repo:  HF dataset repo with the fact pool
                         (``msmarco_encoded.pt`` / ``para_targets.pt``).
            revision:    HF revision. Defaults to the tag matching this
                         package version (e.g. ``v0.1.1``) so code and weights
                         can never mismatch. Pass ``"main"`` to track latest.
            device:      ``"auto"`` (cuda > mps > cpu), or explicit.
            pool_device: Optionally keep the fact pool resident on this device
                         (e.g. ``"cuda"``) instead of streaming CPU->GPU chunks
                         per query. Identical results, more memory.
            chunk_size:  Sentences per scoring chunk during retrieval.
            alpha:       Default selection threshold for ``ask()``: keep
                         retrieved sentence i iff score_i >= alpha * score_1
                         (top-1 always kept; <= 0 keeps all k).
        """
        import json

        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

        rev = revision if revision is not None else _default_revision()
        dev = _resolve_device(device)

        try:
            retriever_path = hf_hub_download(model_repo, RETRIEVER_FILE, revision=rev)
            composer_path = hf_hub_download(model_repo, COMPOSER_FILE, revision=rev)
            encoded_path = hf_hub_download(index_repo, ENCODED_FILE, repo_type="dataset", revision=rev)
            para_path = hf_hub_download(index_repo, PARA_TARGETS_FILE, repo_type="dataset", revision=rev)
        except (RevisionNotFoundError, RepositoryNotFoundError, EntryNotFoundError) as e:
            raise RuntimeError(
                f"Could not fetch Atlas artifacts at revision {rev!r} "
                f"(model={model_repo}, index={index_repo}).\n"
                f"If you installed a development version, try "
                f"Atlas.from_pretrained(..., revision='main').\n"
                f"Original error: {e}"
            ) from e

        # --- models ---
        print(f"  Loading retriever from {retriever_path}...")
        r_ckpt = torch.load(retriever_path, map_location=dev, weights_only=False)
        query_enc = QueryEncoder().to(dev)
        query_enc.load_state_dict(r_ckpt["query_enc"])
        query_enc.eval()
        scorer = DotProductScorer().to(dev)
        scorer.load_state_dict(r_ckpt["scorer"])
        scorer.eval()

        print(f"  Loading composer from {composer_path}...")
        c_ckpt = torch.load(composer_path, map_location=dev, weights_only=False)
        composer = Composer().to(dev)
        composer.load_state_dict(c_ckpt["composer"])
        composer.eval()

        # --- component manifest (descriptive provenance; optional) ---
        manifest = None
        try:
            manifest_path = hf_hub_download(model_repo, MANIFEST_FILE, revision=rev)
            with open(manifest_path) as f:
                manifest = json.load(f)
            comps = manifest.get("components", {})
            comp_str = "  ".join(f"{name}={c.get('version', '?')}" for name, c in comps.items())
            print(f"  Components: {comp_str}")
        except EntryNotFoundError:
            pass  # pre-manifest releases (v0.1.0)

        # --- fact index (kept on CPU unless pool_device is set) ---
        print("  Loading fact index (this can take a minute)...")
        data = torch.load(encoded_path, map_location="cpu", weights_only=False)
        pt = torch.load(para_path, map_location="cpu", weights_only=False)
        data["para_targets"] = pt["para_targets"]
        # Training-only tensors; not used at inference. Freed to save RAM
        # (no effect on any output).
        data.pop("q_embs", None)
        pt.pop("first3_indices", None)
        pt.pop("firstK_indices", None)

        if pool_device is not None:
            pd = torch.device(pool_device)
            data["sent_embs"] = data["sent_embs"].to(pd)
            data["para_targets"] = data["para_targets"].to(pd)

        print(
            f"  Atlas ready. Passages: {len(data['questions'])}  "
            f"Sentences: {data['sent_embs'].shape[0]}  Device: {dev}"
        )

        sonar = Sonar(encoder_model, decoder_model, device=dev)
        atlas = cls(query_enc, scorer, composer, data, sonar, dev, chunk_size, alpha)
        atlas.manifest = manifest
        return atlas

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _retrieve_topk(self, q_emb: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k fact sentences for one query embedding ``[1, D]``.

        Chunked full-pool cosine scan; GPU memory stays bounded regardless of
        pool size. Returns ``(scores [k], indices [k])`` on CPU.
        """
        y = self.query_enc(q_emb.to(self.device))
        y_n = F.normalize(y, dim=-1)
        S = self.sent_embs.shape[0]

        best_scores = torch.full((k,), float("-inf"), device=self.device)
        best_indices = torch.zeros(k, dtype=torch.long, device=self.device)

        for start in range(0, S, self.chunk_size):
            end = min(start + self.chunk_size, S)
            chunk = self.sent_embs[start:end].to(self.device, non_blocking=True)
            chunk_n = F.normalize(chunk, dim=-1)
            sims = (y_n @ chunk_n.T).squeeze(0)

            chunk_idx = torch.arange(start, end, device=self.device)
            combined_scores = torch.cat([best_scores, sims])
            combined_indices = torch.cat([best_indices, chunk_idx])
            top_scores, top_pos = combined_scores.topk(k, largest=True)
            best_scores = top_scores
            best_indices = combined_indices[top_pos]

        return best_scores.cpu(), best_indices.cpu()

    @torch.no_grad()
    def _nearest_passages(
        self,
        composed_emb: torch.Tensor,
        n: int,
        chunk_size: int = 8192,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The n stored passages whose para-embedding is closest to ``composed_emb``."""
        c_n = F.normalize(composed_emb.unsqueeze(0).to(self.device), dim=-1)
        N = self.para_targets.shape[0]
        best_sims = torch.full((n,), float("-inf"), device=self.device)
        best_indices = torch.zeros(n, dtype=torch.long, device=self.device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = F.normalize(self.para_targets[start:end].to(self.device), dim=-1)
            sims = (c_n @ chunk.T).squeeze(0)
            chunk_idx = torch.arange(start, end, device=self.device)
            combined_sims = torch.cat([best_sims, sims])
            combined_indices = torch.cat([best_indices, chunk_idx])
            top_sims, top_pos = combined_sims.topk(n, largest=True)
            best_sims = top_sims
            best_indices = combined_indices[top_pos]

        return best_sims.cpu(), best_indices.cpu()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def retrieve(self, question: str, k: int = 4) -> List[Tuple[str, float]]:
        """Retrieve the top-k fact sentences for a question (no composition)."""
        q_emb = self.sonar.encode(question)
        top_scores, top_indices = self._retrieve_topk(q_emb, k)
        return [
            (self.sentences[i], s)
            for i, s in zip(top_indices.tolist(), top_scores.tolist())
        ]

    @torch.no_grad()
    def ask(
        self,
        question: str,
        k: int = 4,
        alpha: Optional[float] = None,
        n_neighbors: int = 5,
        decode: bool = True,
        score_mode: Optional[str] = None,
    ) -> AtlasResult:
        """Full pipeline: encode -> retrieve top-k -> select -> compose -> decode.

        Args:
            question:    Natural-language question.
            k:           Number of fact sentences to retrieve.
            alpha:       Selection threshold: keep sentence i iff
                         score_i >= alpha * score_1 (top-1 always kept;
                         <= 0 keeps all k). Defaults to the instance
                         ``alpha`` set at load time.
            n_neighbors: Stored passages to report near the composed vector.
            decode:      If False, skip SONAR decoding (``answer`` is None).
            score_mode:  Deprecated and ignored. The composer no longer takes
                         scores; retrieval scores only gate selection (alpha).
        """
        if score_mode is not None:
            import warnings

            warnings.warn(
                "score_mode is deprecated and ignored: the composer no longer "
                "takes scores. Use alpha to control fact selection instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        a = self.alpha if alpha is None else alpha

        # Encode + retrieve
        q_emb = self.sonar.encode(question)
        top_scores, top_indices = self._retrieve_topk(q_emb, k)

        # Select by the relative drop-off rule, then compose the survivors.
        # Scores are used only for this selection; they never enter the
        # composer.
        if a > 0:
            keep = top_scores >= a * top_scores[0]
            keep[0] = True
        else:
            keep = torch.ones_like(top_scores, dtype=torch.bool)

        z_kept = self.sent_embs[top_indices[keep]].unsqueeze(0).to(self.device)
        composed = self.composer(z_kept).squeeze(0).cpu()

        # Grounding: nearest stored passages to the composed vector
        near_sims, near_idxs = self._nearest_passages(composed, n=n_neighbors)
        passages = []
        for sim, ni in zip(near_sims.tolist(), near_idxs.tolist()):
            s, e = self.sent_ranges[ni]
            passages.append({
                "question": self.questions[ni],
                "sentences": self.sentences[s:e],
                "similarity": sim,
                "index": ni,
            })

        answer = self.sonar.decode(composed) if decode else None

        return AtlasResult(
            question=question,
            answer=answer,
            retrieved=[
                (self.sentences[i], s)
                for i, s in zip(top_indices.tolist(), top_scores.tolist())
            ],
            retrieved_indices=top_indices.tolist(),
            kept=keep.tolist(),
            passages=passages,
            embedding=composed,
        )
