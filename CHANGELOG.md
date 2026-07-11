# Changelog

## v0.1.1 - 2026-07-11

New composer (c0.1.1), retriever unchanged (r0.1.0).

- **Variable-K composer.** Retrained on an expanded corpus (MS-MARCO first-K,
  DiscoFuse, reverse WikiSplit, identity) and handles 1 to 4 input facts;
  v0.1.0 was fixed K=3. Gold-input composer cosine at matched K=3:
  0.810 to 0.821. The larger gain is capability: K=2 composes at 0.87,
  and retrieval now feeds up to 4 facts (default `k=4`).
- **Score conditioning removed.** Retrieval scores no longer enter the
  composer at all. They only gate which retrieved facts are composed, via
  the new `alpha` selection rule: keep sentence i iff
  `score_i >= alpha * score_1` (top-1 always kept; `alpha <= 0` keeps all).
  `alpha` is exposed on `from_pretrained()` / `ask()` / `--alpha`.
  `score_mode` is deprecated and ignored.
- **Component manifest.** `manifest.json` ships with the weights and records
  per-component versions (`retriever r0.1.0`, `composer c0.1.1`,
  `index 0.1.1`); surfaced by `from_pretrained()` and `atlas version`.
- **Index.** Sentence pool unchanged; `para_targets.pt` rebuilt over the
  first 4 sentences (was first 3) to match the K=4 default.
- `AtlasResult.kept` reports which retrieved facts survived selection.
- Full-pipeline (val, n=200, alpha=0): composer cosine 0.421 vs top-1
  baseline 0.301; retrieval recall@4 0.525, MRR 0.382. Delta and win-rate
  metrics are retriever-dependent; the composer-quality headline is the
  gold-input cosine above.

## v0.1.0 — 2026-07-04

First public release.

- Retrieval + composition pipeline operating directly in raw SONAR embedding
  space (no intermediate latent space).
- Pretrained retriever (query encoder + cosine scorer, InfoNCE-trained with
  hard negatives) over a ~1.2M-sentence fact pool built from 400k MS MARCO
  passages.
- Pretrained composer (4-layer transformer encoder) producing paragraph-level
  SONAR embeddings from the retrieved top-K, decodable to text via the SONAR
  decoder.
- `Atlas.from_pretrained()` — one-call download of weights + fact index from
  the Hugging Face Hub, revision-pinned to the package version.
- CLI: `atlas ask`, `atlas repl`, `atlas serve` (Gradio demo).
- Inference only; training code is not included in this repository.
