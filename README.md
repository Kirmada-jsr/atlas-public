# Atlas

**Retrieval and composition in SONAR embedding space.**

Atlas is a modular alternative to monolithic language models: instead of one
network doing everything through next-token prediction, Atlas separates the
pipeline into components that operate directly on
[SONAR](https://github.com/facebookresearch/SONAR) sentence embeddings:

1. **Retrieve** — a trained query encoder scores your question against a
   pool of ~1.2M fact-sentence embeddings and picks the top-K.
2. **Compose** — a trained composer folds those K sentence embeddings into a
   single paragraph-level SONAR embedding.
3. **Decode** — the SONAR decoder turns the composed embedding back into text.

No token-level generation model is involved anywhere: the "thinking" happens
entirely in embedding space.

## Quickstart

```bash
pip install "atlas-sonar @ git+https://github.com/Kirmada-jsr/atlas-public@v0.1.0"
atlas ask "what is dobutamine used for?"
```

or in Python:

```python
from atlas import Atlas

atlas = Atlas.from_pretrained()
result = atlas.ask("what is dobutamine used for?")

result.answer      # decoded composed paragraph
result.retrieved   # [(sentence, score), ...] top-K retrieved facts
result.passages    # nearest stored passages to the composed vector (grounding)
result.embedding   # the composed SONAR vector itself
```

`atlas repl` gives you an interactive loop; `atlas serve` launches a local
Gradio demo
(`pip install "atlas-sonar[demo] @ git+https://github.com/Kirmada-jsr/atlas-public@v0.1.0"`).

> **First run downloads ~6.5 GB** (weights ~55 MB, fact index ~5 GB, SONAR
> encoder/decoder ~1.2 GB), cached under `~/.cache/huggingface`. Every later
> run loads from cache.

## Hardware

| Resource | Requirement |
|----------|-------------|
| Disk     | ~7 GB (one-time, cached) |
| RAM      | ~8 GB (the fact pool lives in CPU memory) |
| GPU      | Optional but recommended — decoding is slow on CPU |
| Python   | 3.10+ |

## Knobs

Everything is a defaulted parameter — no source editing needed to experiment:

| Parameter     | Default     | Meaning |
|---------------|-------------|---------|
| `k`           | `3`         | how many fact sentences to retrieve |
| `score_mode`  | `"uniform"` | composer score conditioning; `"uniform"` matches how the composer was trained (recommended), `"retriever"` feeds raw retrieval scores instead |
| `n_neighbors` | `5`         | stored passages reported near the composed vector |
| `decode`      | `True`      | set `False` to skip text decoding (retrieval + embedding only) |
| `device`      | `"auto"`    | `cuda` > `mps` > `cpu` |
| `pool_device` | `None`      | keep the fact pool resident on a device (e.g. `"cuda"`) instead of streaming chunks |
| `revision`    | package tag | HF revision of weights+index; pinned to the installed version so code and weights always match |

## What Atlas is (and isn't)

Atlas answers from a **fixed fact memory** (currently built from MS MARCO
passages). It is a retrieval-composition system, not a free-form generator:
questions outside its memory return the nearest thing it knows. Answers are
grounded — every result shows the exact retrieved sentences and the nearest
stored passages so you can see *why* it answered what it did.

## Scope of this repository

This repository contains the **inference stack only**. The training code
(data pipeline, retriever/composer training, hard-negative mining, evaluation
harness) is intentionally not published here while the underlying research is
still evolving. Model weights and the fact index are released on the
Hugging Face Hub and are downloaded automatically.

## Versioning

Package version, git tag, and Hugging Face artifact revision move together:
installing from the git tag `vX.Y.Z` always loads the weights tagged
`vX.Y.Z` — code and weights cannot drift apart. See
[CHANGELOG.md](CHANGELOG.md).

## Technical report

The full technical report for each major release — architecture with tensor
shapes, the end-to-end inference path, training procedure, and measured
performance — lives in [release-reports/](release-reports/). Current:
[ATLAS v0.1.0](release-reports/ATLAS_v0.1.0_Technical_Report.md).

## Citation

See [CITATION.cff](CITATION.cff), or click "Cite this repository" on GitHub.

## License

Apache-2.0. See [LICENSE](LICENSE).
