# Atlas

**Retrieval, grouping and fusion in SONAR embedding space.**

Atlas is a modular alternative to monolithic language models: instead of one
network doing everything through next-token prediction, Atlas separates the
pipeline into components that operate directly on
[SONAR](https://github.com/facebookresearch/SONAR) sentence embeddings:

1. **Retrieve**: a trained query encoder scores your question against a pool
   of ~8.6M fact-sentence embeddings and picks the top-k.
2. **Dedup**: near-duplicate facts collapse under a plain cosine threshold.
3. **Group**: a trained pairwise classifier decides which of the distinct
   facts can share one fluent sentence, clustering them into groups of up
   to 3.
4. **Fuse**: a trained set transformer folds each group into a single
   SONAR embedding.
5. **Decode**: the SONAR decoder turns each fused embedding back into text;
   the sentences are joined into one answer paragraph.

No token-level generation model is involved anywhere: the "thinking" happens
entirely in embedding space.

## Quickstart

```bash
pip install "atlas-sonar @ git+https://github.com/Kirmada-jsr/atlas-public@v0.2.0"
atlas ask "what is dobutamine used for?"
```

or in Python:

```python
from atlas import Atlas

atlas = Atlas.from_pretrained()
result = atlas.ask("what is dobutamine used for?")

result.answer      # the answer paragraph (one fused sentence per fact group)
result.sentences   # the per-group sentences individually
result.retrieved   # [(sentence, score), ...] top-k retrieved facts
result.deduped     # the distinct facts that survived cosine dedup
result.groups      # how the distinct facts grouped for fusion
result.embeddings  # the fused SONAR vectors, one per group
```

`atlas ask "..." --mode verbose` shows every pipeline stage. `atlas repl`
gives you an interactive loop (toggle verbose live with `:v`); `atlas serve`
launches a local Gradio demo
(`pip install "atlas-sonar[demo] @ git+https://github.com/Kirmada-jsr/atlas-public@v0.2.0"`).

> **First run downloads ~38 GB** (weights ~230 MB, fact pool ~37 GB, SONAR
> encoder/decoder ~1.2 GB), cached under `~/.cache/huggingface`. Every later
> run loads from cache.

## Hardware

| Resource | Requirement |
|----------|-------------|
| Disk     | ~40 GB (one-time, cached) |
| RAM      | ~40 GB (the fact pool lives in CPU memory as fp32) |
| GPU      | Recommended. Retrieval scans the pool per question and SONAR decoding is slow on CPU |
| Python   | 3.10+ |

## Knobs

Everything is a defaulted parameter, no source editing needed to experiment:

| Parameter         | Default     | Meaning |
|-------------------|-------------|---------|
| `k`               | `8`         | how many fact sentences to retrieve |
| `dedup_threshold` | `0.5`       | cosine similarity at or above which two retrieved facts collapse as near-duplicates |
| `fuse_threshold`  | `0.5`       | fusability probability at or above which two distinct facts may share one output sentence |
| `max_group`       | `3`         | maximum facts fused into one sentence |
| `decode`          | `True`      | set `False` to skip text decoding (retrieval + fused embeddings only) |
| `device`          | `"auto"`    | `cuda` > `mps` > `cpu` |
| `pool_device`     | `None`      | keep the fact pool resident on a device (e.g. `"cuda"`) instead of streaming chunks |
| `revision`        | package tag | HF revision of weights+pool; pinned to the installed version so code and weights always match |

## What Atlas is (and isn't)

Atlas answers from a **fixed fact memory** of ~8.6M sentences built from open
QA corpora, plus a small set of identity facts so it can answer questions
about itself and its creator. It is a retrieval-composition system, not a
free-form generator: questions outside its memory return the nearest thing it
knows. Answers are grounded; verbose mode shows the exact retrieved
sentences, the distinct facts after dedup, and which facts fused into each
output sentence, so you can see *why* it answered what it did.

## Scope of this repository

This repository contains the **inference stack only**. The training code
(data pipeline, retriever/grouping/fusion training, hard-negative mining,
LLM distillation of fusion targets, evaluation harness) is intentionally not
published here while the underlying research is still evolving. Model weights
and the fact pool are released on the Hugging Face Hub and are downloaded
automatically.

## Versioning

Package version, git tag, and Hugging Face artifact revision move together:
installing from the git tag `vX.Y.Z` always loads the weights tagged
`vX.Y.Z`, so code and weights cannot drift apart. See
[CHANGELOG.md](CHANGELOG.md).

## Technical report

The full technical report for each major release, covering architecture with
tensor shapes, the end-to-end inference path, training procedure, and
measured performance, lives in [release-reports/](release-reports/). Current:
[ATLAS v0.2.0](release-reports/ATLAS_v0.2.0_Technical_Report.md). Previous:
[ATLAS v0.1.0](release-reports/ATLAS_v0.1.0_Technical_Report.md).

## Citation

See [CITATION.cff](CITATION.cff), or click "Cite this repository" on GitHub.

## License

Apache-2.0. See [LICENSE](LICENSE).
