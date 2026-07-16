# ATLAS v5.4: Technical Report

**Reference specification** for ATLAS v5.4, the build released publicly as
`atlas-public` **v0.2.0**. It documents what the system does, its exact
architecture (with tensor shapes at every stage), the end-to-end path from a
question to an answer, the training procedure for each learned component, the
mechanisms that make it work, and measured performance.

Scale as measured on the released artifacts: **2,102,601 passages**,
**8,615,480 sentences** in the fact pool, SONAR embedding dimension **1024**.
Total trainable parameters **57,209,857**.

This release supersedes v0.1.0 (documented in
[ATLAS v0.1.0](ATLAS_v0.1.0_Technical_Report.md)). The single K-to-1 composer
is replaced by a two-head adaptive-cardinality composer, the pool grows 8.2x,
and the retriever is retrained. Section 9 summarizes the differences.

---

## 1. What it does

ATLAS answers a natural-language question by **retrieving**, **grouping** and
**fusing** sentence embeddings in
[SONAR](https://github.com/facebookresearch/SONAR) space, never generating
tokens autoregressively. Concretely:

- **Retrieve**: score the question against the full sentence pool and take the
  top-K=8 most relevant fact sentences.
- **Dedup**: collapse near-duplicates among those 8 with a cosine threshold,
  leaving M distinct facts.
- **Group**: a learned pairwise classifier decides which of the M distinct
  facts can share one fluent sentence, clustering them into G groups of at
  most 3.
- **Fuse**: a learned set transformer folds each group into one SONAR vector.
- **Decode**: the SONAR decoder turns each fused vector into one sentence; the
  G sentences are joined into the answer paragraph.

The defining property is **adaptive cardinality**: the number of output
sentences is decided by the content, not fixed in advance. v0.1.0 always
collapsed K facts into exactly one vector, which garbled any answer spanning
more than one distinct fact (Section 7).

Capabilities exposed by the system:

| Capability | Entry point |
|---|---|
| Full pipeline: question to answer text | `ask(...)`, `atlas ask` |
| Retrieval only (top-K sentences + scores) | `retrieve(...)` |
| Fusion without decode (return the vectors) | `ask(..., decode=False)` |
| Per-group sentences, ungrouped | `AtlasResult.sentences` |
| Provenance: what survived dedup, and how it grouped | `AtlasResult.deduped`, `.groups` |
| Stage-by-stage trace | `atlas ask --mode verbose`, `:v` in the repl |
| Interactive use | `atlas repl` |

ATLAS does not update its knowledge at inference time. Answers are composed
solely from the fixed fact pool. Out-of-corpus questions return the nearest
thing the pool contains.

**No language model runs at inference.** An LLM is used once, offline, to
distill fusion targets into training data (Section 5.4); the shipped system
contains no LLM and never calls one.

---

## 2. Representation space

Everything operates on raw **SONAR** vectors, dimension **D = 1024**. There is
no FactEncoder, bridge, or intermediate latent space: the retriever scores
SONAR vectors directly, and the fusion model outputs a SONAR vector directly.

- **Encoder**: `cointegrated/SONAR_200_text_encoder` (an `M2M100Encoder`),
  pooled by **attention-masked mean** over the last hidden state,
  `max_length=64`.
- **Decoder**: `raxtemur/SONAR_200_text_decoder`
  (`M2M100ForConditionalGeneration` + `NllbTokenizer`), beam search
  (`num_beams=4`, `no_repeat_ngram_size=3`), language forced to `eng_Latn`.
- Both are frozen; ATLAS never fine-tunes SONAR.

SONAR embeddings have small norm (measured mean **0.24** on fusion targets),
which is why the fusion training loss constrains magnitude separately
(Section 5.6). Cosine similarity is scale-blind, so without an explicit
magnitude term the model is free to drift to the wrong norm and the decoder
garbles (Section 6).

---

## 3. Architecture

Three trainable modules, all downstream of the frozen SONAR encoder. Total
trainable parameters **57,209,857** (retriever 3,150,848 + grouping 525,313 +
fusion 53,533,696).

### 3.1 Retriever

**QueryEncoder**: `3,150,848` params. Maps a question SONAR embedding to a
query vector in the same space. Architecture unchanged from v0.1.0; weights
retrained on the v5.4 pool and then identity fine-tuned (Sections 5.2, 5.3).

```
Input  x : [B, 1024]
  LayerNorm(1024)                 -> [B, 1024]
  Linear(1024 -> 1024) + GELU     -> [B, 1024]
  Linear(1024 -> 1024) + GELU     -> [B, 1024]
  Linear(1024 -> 1024)            -> [B, 1024]
  L2-normalize (dim=-1)           -> [B, 1024]   (unit sphere)
Output y : [B, 1024]
```

Scoring is a plain cosine similarity against the L2-normalized pool. The
v0.1.0 `DotProductScorer` (a single learned InfoNCE temperature) is a
training-time artifact only and is **not part of the inference path** in
v0.2.0; the released checkpoint still carries the parameter, unused.

### 3.2 Grouping model: `525,313` params

Symmetric pairwise **fusability** classifier over two fact embeddings. It
answers "can these two distinct facts share one fluent sentence?", which is
not the same question as "are these two facts near-duplicates" (that one is
answered by a cosine threshold, Section 6).

```
Inputs a, b : [B, 1024]   (two fact vectors)

  phi (shared, applied to each):                       (328,192 params)
      Linear(1024 -> 256) + GELU     -> [B, 256]
      Linear(256 -> 256)             -> [B, 256]
  pa = phi(a) : [B, 256]        pb = phi(b) : [B, 256]

  pair = cat([pa + pb, pa * pb, |pa - pb|], dim=-1)
                                     -> [B, 768]
  cls:                                                 (197,121 params)
      Linear(768 -> 256) + GELU      -> [B, 256]
      Linear(256 -> 1)               -> [B, 1]
      squeeze(-1)                    -> [B]
Output logit : [B]
```

The `[sum, product, abs-diff]` combination is what makes the score symmetric
in `(a, b)`: every term is invariant under swapping the two inputs, so
`score(a,b) == score(b,a)` by construction rather than by training. The
pipeline applies `sigmoid` and thresholds at 0.5.

### 3.3 Fusion model: `53,533,696` params

Set transformer over a group of at most 3 fact vectors. Configuration:
`d_model=1024`, `nhead=8`, `num_layers=4`, `dim_feedforward=4096`,
`activation=gelu`, `dropout=0.0`, `batch_first=True`, **`norm_first=True`**
(pre-LayerNorm).

```
x : [B, 3, 1024]   (group members, zero-padded to 3 slots)
m : [B, 3]         (True = real member, False = padding)

  inp: Linear(1024 -> 1024)          -> [B, 3, 1024]    (1,049,600 params)

  TransformerEncoder (4 layers, pre-LN)                 (50,384,896 params)
      src_key_padding_mask = ~m      -> [B, 3, 1024]
      (padded slots attend to nothing and are never attended to)

  mask-aware mean pool over the 3 slots:
      pooled = (h * m.unsqueeze(-1)).sum(1)
               / m.sum(1, keepdim=True).clamp(min=1)
                                     -> [B, 1024]

  out:                                                  (2,099,200 params)
      Linear(1024 -> 1024) + GELU    -> [B, 1024]
      Linear(1024 -> 1024)           -> [B, 1024]
Output : [B, 1024]   (raw SONAR space, not normalized)
```

Pre-LayerNorm is a deliberate choice for this depth: the post-LN recipe that
served the 2-layer model was unstable when scaled to 4 layers at width 1024
(Section 6).

---

## 4. Inference: question to answer, shape by shape

Batch `B = 1` for a single question. Pool size `S = 8,615,480` sentences.
`sent_embs : [S, 1024]` (fp32, ~32.9 GiB) lives in CPU memory by default;
scoring streams chunks (`chunk_size = 200,000`) to the compute device, so
device memory stays bounded regardless of pool size. Setting `pool_device`
keeps the pool resident instead; results are identical.

**Step 1: encode the question (SONAR).**
```
question (str)
  tokenize (truncate max_length=64)   -> input_ids [1, T], attention_mask [1, T]
  M2M100Encoder                       -> last_hidden_state [1, T, 1024]
  attention-masked mean pool over T   -> q_emb [1, 1024]
```

**Step 2: encode the query (QueryEncoder).**
```
q_emb [1, 1024] -> QueryEncoder -> y [1, 1024]   (L2-normalized)
                -> cast to fp16  -> y [1, 1024]   (fp16)
```

**Step 3: retrieve top-K (chunked full-pool cosine, fp16).**
```
best_scores  [8]  init -1e4 (fp16)      best_indices [8]  init 0 (int64)

for each chunk c of sent_embs ([chunk, 1024]):
    c16   = c.to(device, fp16)                          -> [chunk, 1024]
    c_n   = c16 / c16.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                                                        -> [chunk, 1024]
    sims  = (y @ c_n.T).squeeze(0)                      -> [chunk]
    cat_s = cat([best_scores, sims])                    -> [8 + chunk]
    cat_i = cat([best_indices, arange(start, end)])     -> [8 + chunk]
    best_scores, pos = cat_s.topk(8);  best_indices = cat_i[pos]

=> top_scores  [8]   (raw cosine similarities)
   top_indices [8]   (sentence indices into the pool)
```
`y` is already L2-normalized by the QueryEncoder, so the product is a true
cosine. The scan runs in fp16 on every device (including CPU) so results do
not depend on hardware.

**Step 4: dedup (cosine, no learned head).**
```
E    = normalize(sent_embs[top_indices].float())   -> [8, 1024]
C    = E @ E.T                                     -> [8, 8]

greedy walk in retrieval order over i = 0..7:
    join the first cluster whose EVERY member m satisfies C[i][m] >= 0.5
    otherwise open a new cluster
keep each cluster's first member (the highest-ranked one)

=> deduped_indices [M]     (M <= 8 distinct facts, retrieval order preserved)
```

**Step 5: group (learned fusability).**
```
E = sent_embs[deduped_indices].float()             -> [M, 1024]

for every pair i < j  (M*(M-1)/2 pairs):
    logit = grouping(E[i:i+1] [1,1024], E[j:j+1] [1,1024])   -> [1]
    P[i][j] = P[j][i] = sigmoid(logit)                       (scalar)
P : [M, M]   (symmetric, diagonal unused)

greedy walk over i = 0..M-1:
    join the first group that has < 3 members and whose EVERY member m
      satisfies P[i][m] >= 0.5
    otherwise open a new group

=> groups : G lists of positions into deduped_indices, each of length 1..3
```

**Step 6: fuse each group.**
```
for each group (n = len(group) <= 3):
    embs = sent_embs[group_indices].float()        -> [n, 1024]
    x    = zeros(1, 3, 1024);  x[0, :n] = embs     -> [1, 3, 1024]
    m    = zeros(1, 3, bool);  m[0, :n] = True     -> [1, 3]
    fused = fusion(x, m)[0]                        -> [1024]

=> G fused vectors, one per group
```

**Step 7: decode each fused vector (SONAR decoder).**
```
fused [1024]
  unsqueeze(0)              -> [1, 1024]
  to(device, fp32).unsqueeze(1)
                            -> encoder_outputs.last_hidden_state [1, 1, 1024]
  decoder.generate(num_beams=4, no_repeat_ngram_size=3,
                   forced_bos=eng_Latn, max_length=64)
  batch_decode              -> sentence (str)
```
Each fused vector is injected as a **single-position** encoder memory. The
frozen decoder emits approximately one sentence from one position, which is
precisely why cardinality is handled by grouping upstream rather than by
asking the decoder to enumerate facts from one overloaded vector (Section 6).

**Step 8: assemble the answer.**
```
answer = " ".join(sentences)          (G sentences, in group order)
```

**Result** (`AtlasResult`): `answer` (str), `sentences` (G per-group strings),
`retrieved` (8 sentence/score pairs, raw scores), `retrieved_indices` [8],
`deduped` (M strings), `deduped_indices` [M], `groups` (G index lists),
`embeddings` ([G, 1024] fused vectors).

---

## 5. Training procedure

Every component is trained independently and coupled only at inference. No
component is trained on another's output: the grouping and fusion models see
LLM-distilled fact sets, never retriever output.

### 5.1 Fact pool construction

Three lanes, assembled into one pool:

| Lane | Passages | Source |
|---|---|---|
| PAQ (Wikipedia) | 1,712,591 | PAQ source passages (7,006,992 distinct, sampled) |
| MS-MARCO | 333,004 | carried from the v0.1.0 build (360,710 passages, refiltered) |
| QA golds | 57,006 | NQ / TriviaQA / SQuAD gold passages |
| **Total** | **2,102,601** | **8,615,405 sentences** |

- Sentence segmentation: NLTK Punkt.
- Filters: 4 to 60 words per sentence; 2 to 5 sentences per passage (v0.1.0
  used 2 to 4; the wider bound was staged for later work).
- Cleanup at build: exact-match blocklist for boilerplate (2,030 sentences
  dropped, e.g. "From Wikipedia, the free encyclopedia") and exact-duplicate
  collapse (271,056 sentences collapsed).
- Every sentence is SONAR-encoded to `sent_embs [8615405, 1024]`.
- A small set of identity sentences (Section 5.3) is appended at load, giving
  the released pool **8,615,480** sentences.

**Query table** (retriever training pairs), 2,947,148 total:

| Source | Pairs |
|---|---|
| MS-MARCO | 333,004 |
| NQ | 30,644 |
| TriviaQA | 31,253 |
| SQuAD | 52,247 |
| PAQ | 2,500,000 |
| **Train / Val** | **2,946,148 / 1,000** |

Validation is **real QA only** (never PAQ), 1,000 held-out queries.

### 5.2 Retriever (InfoNCE)

Recipe carried from the validated v0.1.0 retriever training, scaled to the
larger pool.

```
q_batch [B, 1024] -> QueryEncoder -> y_batch [B, 1024]
candidates = [ in-batch sentences | mined hard negatives ]
scaled_scores = pairwise(candidates, y_batch) * temperature   -> [B, P_all]
loss = InfoNCE(scaled_scores, pos_mask, neg_mask)   (sum-of-positives form)
```

- Batch 512, one passage per slot (guarantees cross-passage negatives).
- AdamW, lr 3e-4, weight decay 1e-4, grad clip 5.0, cosine schedule to 1%.
- 60 epochs with **ANCE-style index refresh every 20 epochs**, in-process so
  optimizer state stays continuous.
- Hard negatives premined per query: **1,024 hard + 1,024 random = 2,048
  candidates**, of which **128 are sampled per step**.
- 50/50 real/PAQ mix per epoch: all 446,148 real pairs plus an equal
  deterministic PAQ sample.
- Validation is **full-pool** (Recall@K, MRR, Top-1), not in-batch loss.

The released checkpoint is the **best-val-MRR** epoch, which was **epoch 21**
(MRR 0.3578). Training continued to epoch 60 and did not improve; this
matches the previously documented ANCE-refresh feedback behavior.

### 5.3 Identity fine-tune

The query encoder is lightly fine-tuned to give the system a sense of its own
identity, and a small set of identity sentences is injected into the pool. The
pre-identity retriever is preserved unmodified; the fine-tune writes a
separate checkpoint.

The fine-tune is constrained by an **anchor loss** that pins the encoder's
output on held-out general queries to its frozen original, so the adjustment
stays local to the identity questions instead of reshaping the whole query
space. Measured drift against the original encoder is **0.994 cosine**, i.e.
no measurable forgetting: the retrieval metrics in Section 7.1 carry over to
the released checkpoint.

### 5.4 Fusion target distillation (offline, LLM)

Training targets for both composer heads come from an LLM's judgment about
which facts belong together and how they read when joined. This runs **once,
offline**; no LLM is in the shipped system.

- **Model**: Qwen3-8B, non-thinking mode, bfloat16, served with vLLM,
  temperature 0.0, `max_tokens=900`, `max_model_len=2048`.
- **Input**: for each query, its retrieved top-8 facts, **cosine-deduped
  first** at threshold 0.5 (the same rule inference uses, so the LLM sees
  what the pipeline will see).
- **Prompt contract**: partition the facts into groups of **at most 3**
  (prefer 2); group only if they combine into **one** natural grammatical
  sentence (same entity, or a clear relation: cause, contrast, elaboration,
  apposition, near-duplicate); a fact that does not combine stays alone; each
  group's sentence uses **only** information in that group's facts. Output is
  strict JSON.
- **Records**: each group is saved with its member pool indices, the fused
  sentence, the source id, and the query (set) id, which is what makes both a
  set-level train/val split and the grouping labels recoverable.
- **Filters** applied when materializing tensors: group size <= 3 (the LLM
  occasionally violated the cap), fused text non-empty and ending in terminal
  punctuation (drops pool-fragment artifacts).
- **Targets**: the fused sentence is SONAR-encoded to `Y [N, 1024]`; measured
  target norm mean **0.24**, consistent with raw SONAR.

Yield: **15,593,316 groups** across sources (msmarco 1,987,874; nq 145,578;
trivia 190,095; squad 340,760; paq 12,929,009).

### 5.5 Grouping model (BCE on pairwise fusability)

Labels are **reconstructed from the LLM's partition**: two facts in the same
group are a positive pair, two facts in different groups of the same query are
a negative pair. This is the key trick, since the LLM never emitted pairwise
labels directly.

```
facts       : 3,128,771
positives   :   490,714 pairs   (same group)
negatives   : 1,472,142 pairs   (different group, subsampled to 3 per positive)
split       : 90/10 -> train 1,766,570 / val 196,286
loss        : binary_cross_entropy_with_logits
optimizer   : AdamW, lr 1e-3, weight decay 1e-2, batch 8,192, 15 epochs
```

### 5.6 Fusion model (cosine + magnitude regression)

```
Input  : X [B, 3, 1024] fp32, group member SONAR vectors, zero-padded
         M [B, 3] bool, real-member mask
Target : Y [B, 1024], SONAR of the LLM's fused sentence
Loss   : (1 - cos(pred, Y)).mean()
         + 0.1 * ((||pred|| - ||Y||)^2).mean()      <- magnitude term
         + 0.1 * mse(pred, Y)
```

- **Mix**: `balanced`, all real groups plus an equal PAQ sample:
  **train 5,178,560** (real 2,589,280 + PAQ 2,589,280), **val 76,027**.
  A full-PAQ variant (every PAQ group, no subsample) trains to the same
  quality, so balanced ships.
- **Split**: per-source holdout at **set level** for real sources (every group
  of a held-out query is held out together, so no query leaks across the
  split) and group level for PAQ.
- **Optimizer**: AdamW, lr 1e-3, weight decay 1e-2, batch 4,096, 40 epochs,
  50,560 total steps, **2,000-step warmup then cosine decay**, **grad clip
  1.0**.
- The constant-lr, no-clip recipe that worked for the smaller model diverged
  around epoch 12 at this size; warmup plus clipping fixed it.

---

## 6. What makes it work

- **Adaptive cardinality.** The frozen SONAR decoder emits approximately one
  sentence from a single-position memory. Forcing K unrelated facts into one
  vector therefore does not produce a multi-fact paragraph, it produces
  garble: related facts fuse acceptably, unrelated facts collide. v5.4 stops
  asking the decoder to do the impossible and instead decides **how many
  vectors to produce**. This is the central fix of the release.
- **Division of labor between cosine and learned heads, decided empirically.**
  Redundancy removal is done by a one-line cosine threshold because a learned
  dedup/selection head was Pareto-dominated by cosine on held-out data.
  Fusability is done by a learned head because it beats cosine there (0.8247
  vs 0.7730 AUC). These are different questions: two facts can be highly
  similar and not fusable, or dissimilar and perfectly fusable.
- **The magnitude term is load-bearing.** Cosine is scale-blind, so a
  cosine-only objective let the fusion output drift to roughly 6x the correct
  norm, which the decoder rendered as garbage. Adding the explicit norm
  penalty is what produced coherent decodes; it was the single most important
  fix in the composer's development.
- **Capacity was the real ceiling, and it was tested rather than assumed.**
  A 7.6M fusion model plateaued at val cosine ~0.77. Holding data, loss and
  schedule fixed and scaling to 53.5M cleared the plateau to ~0.84 with no
  overfitting (train 0.871 vs val ~0.84, gap ~0.03). Qualitatively the larger
  model also preserves negation, e.g. rendering "ATLAS is **not** an acronym"
  where the smaller model dropped the "not".
- **Pre-LayerNorm plus warmup plus grad clipping for depth.** Scaling to 4
  layers at width 1024 made the post-LN, constant-lr recipe unstable; pre-LN
  with a 2,000-step warmup and clip 1.0 trains cleanly for 40 epochs.
- **Set-level splits prevent leakage.** Groups derived from the same query go
  to the same side of the train/val split, so val cosine measures
  generalization to unseen queries rather than memorization of a query's other
  groups.
- **Distillation at training time only.** The LLM contributes judgment
  (what groups, what phrasing) into a fixed dataset. At inference the system
  is 57.2M parameters of small heads over frozen SONAR, with no LLM anywhere.
- **Anchor loss makes the identity fine-tune safe.** Fine-tuning a retriever on
  a narrow set of questions would ordinarily damage general retrieval. Pinning
  the encoder to its frozen self on held-out general queries holds drift to
  0.994 cosine while still meeting the fine-tune's objective.

---

## 7. Metrics (validation)

### 7.1 Retriever (full-pool, 1,000 held-out real-QA queries)

Best epoch (21 of 60), the released checkpoint before identity fine-tuning:

| Metric | Value |
|---|---|
| MRR | 0.3578 |
| Recall@3 | 0.4980 |
| Recall@10 | 0.6400 |
| Top-1 | 0.3410 |

> As in v0.1.0, this metric is pessimistic: a retrieved sentence counts only
> if it belongs to the query's own gold passage. With 8.6M sentences drawn
> from overlapping corpora, semantically correct sentences retrieved from a
> different passage score as misses. Recall@3 of 0.4980 is a conservative
> lower bound on practical retrieval quality. For reference, the comparable
> v0.1.0 figure was 0.4550 over a pool 8.2x smaller.

### 7.2 Grouping model (196,286 held-out pairs)

| Model | Fusability AUC |
|---|---|
| **Grouping model (learned)** | **0.8247** |
| Cosine baseline | 0.7730 |

This is the one place in the system where a learned head beat cosine, which
is why it is the only learned head in the selection path.

### 7.3 Fusion model (per-source, held-out)

`val-cos` is cosine to the target fused-sentence embedding. `decode-q` is a
round-trip: decode the fused vector to text, re-encode that text, and measure
cosine to the target. `decode-q` is evaluated on fusions only (groups of 2 or
more).

**Released model (53.5M):**

| Source | val-cos | decode-q | val groups | decoded |
|---|---|---|---|---|
| msmarco | 0.838 | 0.649 | 5,528 | 338 |
| nq | 0.850 | 0.708 | 19,067 | 400 |
| trivia | 0.817 | 0.642 | 24,351 | 233 |
| squad | 0.837 | 0.651 | 26,081 | 188 |
| paq | 0.851 | 0.684 | 1,000 | 191 |
| **mean** | **~0.84** | **~0.67** | | |

Final train cosine 0.871 against val ~0.84: a gap of ~0.03, i.e. no
overfitting at this capacity.

**Capacity comparison (7.6M model, identical data / loss / schedule):**

| Source | val-cos (7.6M) | val-cos (53.5M) | decode-q (7.6M) | decode-q (53.5M) |
|---|---|---|---|---|
| msmarco | 0.770 | 0.838 | 0.577 | 0.649 |
| nq | 0.783 | 0.850 | 0.629 | 0.708 |
| trivia | 0.743 | 0.817 | 0.553 | 0.642 |
| squad | 0.753 | 0.837 | 0.543 | 0.651 |
| paq | 0.786 | 0.851 | 0.628 | 0.684 |
| **mean** | **~0.77** | **~0.84** | **~0.59** | **~0.67** |

7x the parameters buys **+0.072 val-cos** and **+0.081 decode-q** on the mean
(per-source val-cos gains range from +0.065 on PAQ to +0.084 on SQuAD),
confirming the plateau was capacity-bound rather than data-bound.

---

## 8. Configuration reference

| Group | Parameter | Value |
|---|---|---|
| SONAR | encoder / decoder | `cointegrated/SONAR_200_text_encoder` / `raxtemur/SONAR_200_text_decoder` |
| SONAR | dim, lang, enc max_length | 1024, `eng_Latn`, 64 |
| SONAR | decode beams, no_repeat_ngram | 4, 3 |
| Retriever | QueryEncoder hidden | 1024 |
| Retriever | lr, weight decay, grad clip | 3e-4, 1e-4, 5.0 |
| Retriever | batch, epochs, refresh_every | 512, 60, 20 |
| Retriever | hard-neg index (hard / random) | 2,048 (1,024 / 1,024) |
| Retriever | hard-neg sampled per step | 128 |
| Grouping | phi hidden, cls input | 256, 768 (3 x 256) |
| Grouping | lr, weight decay, batch, epochs | 1e-3, 1e-2, 8,192, 15 |
| Grouping | neg per pos | 3 |
| Fusion | heads / layers / d_model / ffn | 8 / 4 / 1024 / 4096 |
| Fusion | dropout, activation, norm_first | 0.0, GELU, True (pre-LN) |
| Fusion | loss | cosine + 0.1 x norm + 0.1 x mse |
| Fusion | lr, weight decay, batch, epochs | 1e-3, 1e-2, 4,096, 40 |
| Fusion | warmup, schedule, grad clip | 2,000 steps, cosine decay, 1.0 |
| Distillation | model, precision, temperature | Qwen3-8B (non-thinking), bf16, 0.0 |
| Data | passages, sentences | 2,102,601 / 8,615,480 |
| Data | sentences/passage, words/sentence | 2 to 5, 4 to 60 |
| Data | fusion groups (total / train / val) | 15,593,316 / 5,178,560 / 76,027 |
| Inference | top-K, dedup TH, fuse TH, max group | 8, 0.5, 0.5, 3 |
| Inference | retrieval chunk, scan precision | 200,000, fp16 |

---

## 9. Changes from v0.1.0

| Aspect | v0.1.0 | v0.2.0 |
|---|---|---|
| Composer | one K-to-1 transformer, 54.8M | two heads: grouping 0.5M + fusion 53.5M |
| Output | one vector, one sentence | one vector per group, G sentences joined |
| Cardinality | fixed | adaptive (content-decided) |
| Retrieval depth | k=3 (v0.1.0) / k=4 (v0.1.1) | k=8 |
| Selection | alpha score-drop rule | cosine dedup + learned fusability grouping |
| Pool | 1,045,350 sentences | 8,615,480 sentences (8.2x) |
| Retriever | v5.1 canonical | retrained + identity fine-tuned |
| Grounding | nearest stored passages | stage-by-stage trace (verbose mode) |
| Scores into composer | uniform 1/K vector | never; scores only rank retrieval |
| Trainable params | ~58.0M | 57,209,857 |
| Footprint | ~7 GB disk, ~8 GB RAM | ~40 GB disk, ~40 GB RAM |

Removed from the public API: `alpha`, `score_mode`, `n_neighbors`, and
`AtlasResult.passages`. Added: `dedup_threshold`, `fuse_threshold`,
`max_group`, and the `sentences` / `deduped` / `groups` / `embeddings` result
fields.

---

The public package is a behavior-preserving port of the development inference
path: an equivalence gate over general and identity questions produces
byte-identical group sentences between the development harness and the
released package, using the same weights and pool.
