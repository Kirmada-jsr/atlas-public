"""Atlas inference pipeline.

``Atlas.from_pretrained()`` downloads the released weights and fact pool
from the Hugging Face Hub (pinned to the revision matching this package
version) and assembles the full ask pipeline:

    question --SONAR encode--> query embedding
             --QueryEncoder + chunked cosine scan--> top-K fact sentences
             --cosine dedup--> distinct facts (near-duplicates collapsed)
             --GroupingModel--> groups of fusable facts (up to 3 per group)
             --FusionModel, per group--> one fused SONAR vector per group
             --SONAR decode, per group--> one sentence per group

The per-group sentences are joined into one answer paragraph. The retrieval,
dedup, grouping and fusion math is ported behavior-for-behavior from the
validated dev inference script; public-facing outputs are identical to it.

v0.2.0: retrieval scores never enter any model. They only rank the pool scan;
which facts are composed together is decided by the cosine dedup and the
learned grouping head. The fact pool includes a small set of Atlas identity
facts, so Atlas can answer questions about itself and its creator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .models import SONAR_DIM, FusionModel, GroupingModel, QueryEncoder
from .sonar import DEFAULT_DECODER, DEFAULT_ENCODER, Sonar

DEFAULT_MODEL_REPO = "kirmada-jsr/atlas"
DEFAULT_INDEX_REPO = "kirmada-jsr/atlas-index"

# Filenames inside the HF repos (written by scripts/upload_to_hf.py).
RETRIEVER_FILE = "retriever.pt"
MODEL_A_FILE = "model_a.pt"
MODEL_B_FILE = "model_b.pt"
POOL_FILE = "pool.pt"
IDENTITY_POOL_FILE = "identity_pool.pt"
MANIFEST_FILE = "manifest.json"

# Validated inference defaults (see ask()).
DEFAULT_K = 8
DEFAULT_DEDUP_THRESHOLD = 0.5
DEFAULT_FUSE_THRESHOLD = 0.5
DEFAULT_MAX_GROUP = 3
DEFAULT_CHUNK_SIZE = 200_000


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
    answer: Optional[str]                      # per-group sentences joined into a paragraph; None if decode=False
    sentences: List[str] = field(default_factory=list)  # one decoded sentence per group
    retrieved: List[Tuple[str, float]] = field(default_factory=list)  # (sentence, retriever cosine score), top-k
    retrieved_indices: List[int] = field(default_factory=list)  # pool indices of `retrieved`
    deduped: List[str] = field(default_factory=list)  # distinct facts after cosine dedup
    deduped_indices: List[int] = field(default_factory=list)  # pool indices of `deduped`
    groups: List[List[int]] = field(default_factory=list)  # index into `deduped`, one list per group
    embeddings: Optional[torch.Tensor] = None  # fused SONAR vectors [G, D], CPU


class Atlas:
    """Pretrained Atlas: retrieval, grouping and fusion over a SONAR fact memory."""

    def __init__(
        self,
        query_enc: QueryEncoder,
        grouper: GroupingModel,
        fuser: FusionModel,
        sentences: List[str],
        sent_embs: torch.Tensor,
        sonar: Sonar,
        device: torch.device,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.query_enc = query_enc
        self.grouper = grouper
        self.fuser = fuser
        self.sentences = sentences
        self.sent_embs = sent_embs        # [S, D]
        self.sonar = sonar
        self.device = device
        self.chunk_size = chunk_size
        self.manifest: Optional[Dict] = None

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
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> "Atlas":
        """Download weights + fact pool from the HF Hub and build the pipeline.

        Args:
            model_repo:  HF model repo with ``retriever.pt`` / ``model_a.pt``
                         / ``model_b.pt``.
            index_repo:  HF dataset repo with the fact pool (``pool.pt`` and
                         ``identity_pool.pt``).
            revision:    HF revision. Defaults to the tag matching this
                         package version (e.g. ``v0.2.0``) so code and weights
                         can never mismatch. Pass ``"main"`` to track latest.
            device:      ``"auto"`` (cuda > mps > cpu), or explicit.
            pool_device: Optionally keep the fact pool resident on this device
                         (e.g. ``"cuda"``) instead of streaming CPU->GPU chunks
                         per query. Identical results, more memory.
            chunk_size:  Sentences per scoring chunk during retrieval.
        """
        import json

        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

        rev = revision if revision is not None else _default_revision()
        dev = _resolve_device(device)

        try:
            retriever_path = hf_hub_download(model_repo, RETRIEVER_FILE, revision=rev)
            model_a_path = hf_hub_download(model_repo, MODEL_A_FILE, revision=rev)
            model_b_path = hf_hub_download(model_repo, MODEL_B_FILE, revision=rev)
            pool_path = hf_hub_download(index_repo, POOL_FILE, repo_type="dataset", revision=rev)
            identity_path = hf_hub_download(index_repo, IDENTITY_POOL_FILE, repo_type="dataset", revision=rev)
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

        print(f"  Loading grouping model from {model_a_path}...")
        a_ckpt = torch.load(model_a_path, map_location=dev, weights_only=False)
        grouper = GroupingModel().to(dev)
        grouper.load_state_dict(a_ckpt["model"])
        grouper.eval()

        print(f"  Loading fusion model from {model_b_path}...")
        b_ckpt = torch.load(model_b_path, map_location=dev, weights_only=False)
        fuser = FusionModel().to(dev)
        fuser.load_state_dict(b_ckpt["model"])
        fuser.eval()

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
            pass

        # --- fact pool (kept on CPU unless pool_device is set) ---
        print("  Loading fact pool (36 GB; this can take a few minutes)...")
        pool = torch.load(pool_path, map_location="cpu", weights_only=False)
        sentences = list(pool["sentences"])
        sent_embs = pool["sent_embs"]
        # Passage bookkeeping tensors are training-only; freed to save RAM
        # (no effect on any output).
        pool.pop("sent_ranges", None)
        pool.pop("passage_lane", None)

        # The identity facts are part of the released memory: always appended.
        idp = torch.load(identity_path, map_location="cpu", weights_only=False)
        sent_embs = torch.cat([sent_embs, idp["embs"].to(sent_embs.dtype)], dim=0)
        sentences = sentences + list(idp["sentences"])

        if pool_device is not None:
            sent_embs = sent_embs.to(torch.device(pool_device))

        print(
            f"  Atlas ready. Sentences: {sent_embs.shape[0]:,}  Device: {dev}"
        )

        sonar = Sonar(encoder_model, decoder_model, device=dev)
        atlas = cls(query_enc, grouper, fuser, sentences, sent_embs, sonar, dev, chunk_size)
        atlas.manifest = manifest
        return atlas

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _retrieve_topk(self, q_emb: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k fact sentences for one query embedding ``[1, D]``.

        Chunked full-pool cosine scan in fp16; device memory stays bounded
        regardless of pool size. Returns ``(scores [k], indices [k])`` on CPU.
        """
        y = self.query_enc(q_emb.to(self.device)).to(torch.float16)
        S = self.sent_embs.shape[0]

        best_scores = torch.full((k,), -1e4, device=self.device, dtype=torch.float16)
        best_indices = torch.zeros(k, dtype=torch.long, device=self.device)

        for start in range(0, S, self.chunk_size):
            end = min(start + self.chunk_size, S)
            chunk = self.sent_embs[start:end].to(self.device, torch.float16)
            chunk = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            sims = (y @ chunk.T).squeeze(0)

            combined_scores = torch.cat([best_scores, sims])
            combined_indices = torch.cat([best_indices, torch.arange(start, end, device=self.device)])
            top_scores, top_pos = combined_scores.topk(k, largest=True)
            best_scores = top_scores
            best_indices = combined_indices[top_pos]

        return best_scores.cpu(), best_indices.cpu()

    def _dedup(self, pool_indices: List[int], threshold: float) -> List[int]:
        """Collapse near-duplicate retrieved facts by cosine similarity.

        Greedy: walk the facts in retrieval order, join the first cluster the
        fact is similar (>= threshold) to every member of, else start a new
        cluster. Returns the position (into ``pool_indices``) of each
        cluster's first member.
        """
        E = F.normalize(self.sent_embs[torch.tensor(pool_indices)].float(), dim=-1)
        C = (E @ E.T).tolist()
        clusters: List[List[int]] = []
        for i in range(len(pool_indices)):
            placed = False
            for cl in clusters:
                if all(C[i][m] >= threshold for m in cl):
                    cl.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
        return [cl[0] for cl in clusters]

    @torch.no_grad()
    def _group(self, pool_indices: List[int], threshold: float, max_group: int) -> List[List[int]]:
        """Cluster distinct facts into fusable groups with the grouping model.

        Greedy over pairwise fusability probabilities: a fact joins the first
        group it is fusable (sigmoid >= threshold) with every member of, if
        that group has room (< max_group); else it starts a new group.
        Returns groups of positions into ``pool_indices``.
        """
        E = self.sent_embs[torch.tensor(pool_indices)].float().to(self.device)
        n = len(pool_indices)
        P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                p = torch.sigmoid(self.grouper(E[i:i + 1], E[j:j + 1])).item()
                P[i][j] = P[j][i] = p

        groups: List[List[int]] = []
        for i in range(n):
            placed = False
            for grp in groups:
                if len(grp) < max_group and all(P[i][m] >= threshold for m in grp):
                    grp.append(i)
                    placed = True
                    break
            if not placed:
                groups.append([i])
        return groups

    @torch.no_grad()
    def _fuse(self, pool_indices: List[int], max_group: int) -> torch.Tensor:
        """Fuse one group of facts into a single SONAR vector ``[D]``."""
        members = pool_indices[:max_group]
        embs = self.sent_embs[torch.tensor(members)].float()
        x = torch.zeros(1, max_group, SONAR_DIM, device=embs.device)
        mask = torch.zeros(1, max_group, dtype=torch.bool, device=embs.device)
        x[0, :len(members)] = embs
        mask[0, :len(members)] = True
        return self.fuser(x.to(self.device), mask.to(self.device))[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def retrieve(self, question: str, k: int = DEFAULT_K) -> List[Tuple[str, float]]:
        """Retrieve the top-k fact sentences for a question (no fusion)."""
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
        k: int = DEFAULT_K,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
        fuse_threshold: float = DEFAULT_FUSE_THRESHOLD,
        max_group: int = DEFAULT_MAX_GROUP,
        decode: bool = True,
    ) -> AtlasResult:
        """Full pipeline: retrieve top-k, dedup, group, fuse and decode.

        Args:
            question:        Natural-language question.
            k:               Number of fact sentences to retrieve.
            dedup_threshold: Cosine similarity at or above which two retrieved
                             facts count as near-duplicates and collapse.
            fuse_threshold:  Fusability probability at or above which two
                             distinct facts may share one output sentence.
            max_group:       Maximum facts fused into one sentence.
            decode:          If False, skip SONAR decoding (``answer`` and
                             ``sentences`` are empty; fused ``embeddings``
                             are still returned).
        """
        q_emb = self.sonar.encode(question)
        top_scores, top_indices = self._retrieve_topk(q_emb, k)
        idxs = top_indices.tolist()

        reps = self._dedup(idxs, dedup_threshold)
        deduped_idx = [idxs[r] for r in reps]

        groups = self._group(deduped_idx, fuse_threshold, max_group)
        fused = [self._fuse([deduped_idx[i] for i in grp], max_group) for grp in groups]

        sentences = [self.sonar.decode(v) for v in fused] if decode else []
        answer = " ".join(sentences) if decode else None

        return AtlasResult(
            question=question,
            answer=answer,
            sentences=sentences,
            retrieved=[
                (self.sentences[i], s)
                for i, s in zip(idxs, top_scores.tolist())
            ],
            retrieved_indices=idxs,
            deduped=[self.sentences[i] for i in deduped_idx],
            deduped_indices=deduped_idx,
            groups=groups,
            embeddings=torch.stack([v.cpu() for v in fused]) if fused else None,
        )
