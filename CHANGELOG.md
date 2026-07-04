# Changelog

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
