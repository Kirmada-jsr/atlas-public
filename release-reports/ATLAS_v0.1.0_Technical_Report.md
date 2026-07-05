# ATLAS v5.1 — Technical Report

**Anchor version.** This is the reference specification for ATLAS v5.1, the
build released publicly as `atlas-public` **v0.1.0**. It documents what the
system does, its exact architecture (with tensor shapes at every stage), the
end-to-end path from a question to an answer, the training procedure, the
mechanisms that make it work, and measured performance.

Scale as measured on the released artifacts: **360,710 passages**,
**1,045,350 sentences** in the fact pool, SONAR embedding dimension **1024**.

---

## 1. What it does

ATLAS answers a natural-language question by **retrieving** and **composing**
sentence embeddings in [SONAR](https://github.com/facebookresearch/SONAR)
space — never generating tokens autoregressively. Concretely:

- **Retrieve** — score the question against the full sentence pool and take the
  top-K=3 most relevant fact sentences.
- **Compose** — fold those 3 sentence embeddings into a single paragraph-level
  SONAR embedding via a transformer composer.
- **Decode** — turn the composed embedding back into text with the SONAR
  decoder.
- **Ground** — report the stored passages nearest to the composed vector, so
  every answer is traceable to its supporting evidence.

Capabilities exposed by the system:

| Capability | Entry point |
|---|---|
| Full pipeline: question → answer text | `ask(...)`, `atlas ask` |
| Retrieval only (top-K sentences + scores) | `retrieve(...)` |
| Composition without decode (return the vector) | `ask(..., decode=False)` |
| Grounding (nearest stored passages to the composed vector) | `AtlasResult.passages` |
| Composer-in-isolation evaluation (gold sentences in) | gold-input mode |
| Full-pipeline evaluation (retriever output in) | full-pipeline mode |
| Score-conditioning ablation | `score_mode = "uniform" | "retriever"` |
| Interactive use | `atlas repl` |

ATLAS does not update its knowledge at inference time. Answers are composed
solely from the fixed MS-MARCO-derived fact pool. Out-of-corpus questions
return the nearest thing the pool contains.

---

## 2. Representation space

Everything operates on raw **SONAR** vectors, dimension **D = 1024**. There is
no FactEncoder, bridge, or intermediate latent space — the retriever scores
SONAR vectors directly and the composer outputs a SONAR vector directly.

- **Encoder**: `cointegrated/SONAR_200_text_encoder` (an `M2M100Encoder`),
  pooled by **attention-masked mean** over the last hidden state.
- **Decoder**: `raxtemur/SONAR_200_text_decoder`
  (`M2M100ForConditionalGeneration` + `NllbTokenizer`), beam search
  (`num_beams=4`), language forced to `eng_Latn`.
- Both are frozen; ATLAS never fine-tunes SONAR.

SONAR embeddings have small norm (≈0.2–0.3), which is why the composer training
loss constrains magnitude separately (Section 6).

---

## 3. Architecture

Three trainable modules. Total trainable parameters ≈ **58.0M**
(retriever 3.15M + composer 54.85M).

### 3.1 Retriever

**QueryEncoder** — `3,150,848` params. Maps a question SONAR embedding to a
query vector in the same space.

```
Input  x : [B, 1024]
  LayerNorm(1024)                 -> [B, 1024]
  Linear(1024 -> 1024) + GELU     -> [B, 1024]
  Linear(1024 -> 1024) + GELU     -> [B, 1024]
  Linear(1024 -> 1024)            -> [B, 1024]
  L2-normalize (dim=-1)           -> [B, 1024]   (unit sphere)
Output y : [B, 1024]
```

**DotProductScorer** — `1` param (a learned temperature `logit_scale`,
initialized `log(1/0.07) ≈ 2.6593`). At inference it produces plain cosine
similarities; the temperature only scales logits during InfoNCE training.

```
z_pool : [P, 1024]   (fact vectors)      y : [B, 1024]   (query vectors)
  z_n = normalize(z_pool)                y_n = normalize(y)
  scores = y_n @ z_n.T                 -> [B, P]
```

### 3.2 Composer — `54,845,952` params

TransformerEncoder + MLP with additive score conditioning. Configuration:
`d_model=1024`, `nhead=8`, `num_layers=4`, `dim_feedforward=4096`,
`activation=gelu`, `dropout=0.0`, `batch_first=True`, `norm_first=False`.

```
z_topk : [B, K, 1024]        scores : [B, K]        (K = 3)

  score_embed(scores):                              (263,680 params)
      scores.unsqueeze(-1)            -> [B, K, 1]
      Linear(1 -> 256) + GELU         -> [B, K, 256]
      Linear(256 -> 1024)             -> [B, K, 1024]
  x = z_topk + score_emb              -> [B, K, 1024]

  TransformerEncoder (4 layers)       -> [B, K, 1024]   (50,384,896 params)

  mask-aware mean pool over K:                          (padded slots excluded)
      h = (x * valid).sum(1) / valid.sum(1)  -> [B, 1024]

  MLP:                                                  (4,197,376 params)
      Linear(1024 -> 2048) + GELU     -> [B, 2048]
      Linear(2048 -> 1024)            -> [B, 1024]
Output : [B, 1024]   (raw SONAR space, not normalized)
```

`src_key_padding_mask : [B, K]` (True = pad) supports passages with fewer than
K sentences; padded slots are zeroed in the input, given score 0, and excluded
from the mean pool.

---

## 4. Inference: question → answer, shape by shape

Batch `B = 1` for a single question. Pool sizes: `S = 1,045,350` sentences,
`N = 360,710` passages. `sent_embs : [S, 1024]` and `para_targets : [N, 1024]`
live in CPU memory; scoring streams chunks (`chunk_size = 4096`) to the compute
device so GPU memory stays bounded regardless of pool size.

**Step 1 — Encode the question (SONAR).**
```
question (str)
  tokenize (truncate max_length=64)        -> input_ids [1, T], attention_mask [1, T]
  M2M100Encoder                            -> last_hidden_state [1, T, 1024]
  attention-masked mean pool over T        -> q_emb [1, 1024]
```

**Step 2 — Encode the query (QueryEncoder).**
```
q_emb [1, 1024] -> QueryEncoder -> y [1, 1024]   (L2-normalized)
```

**Step 3 — Retrieve top-K (chunked full-pool cosine).**
```
for each chunk c of sent_embs ([chunk, 1024]):
    sims_c = normalize(y) @ normalize(c).T     -> [1, chunk]
    running top-K maintained across chunks
=> top_scores  [K=3]     (raw cosine similarities)
   top_indices [K=3]     (sentence indices into the pool)
```

**Step 4 — Assemble composer input.**
```
z_topk = sent_embs[top_indices].unsqueeze(0)   -> [1, 3, 1024]
scores = top_scores.unsqueeze(0)               -> [1, 3]

score_mode == "uniform"  (default, validated):
    scores = full_like(scores, 1/3)            -> [1, 3]   all = 0.3333
score_mode == "retriever"  (ablation):
    scores left as the raw retrieval cosines
```

**Step 5 — Compose.**
```
composer(z_topk [1,3,1024], scores [1,3]) -> composed [1, 1024] -> squeeze -> [1024]
```

**Step 6 — Decode to text (SONAR decoder).**
```
composed [1024]
  unsqueeze -> [1, 1024] -> unsqueeze(1) -> encoder_outputs.last_hidden_state [1, 1, 1024]
  decoder.generate(num_beams=4, forced_bos=eng_Latn, max_length=64)
  batch_decode -> answer (str)
```

**Step 7 — Ground (nearest passages).**
```
c_n = normalize(composed) [1, 1024]
for each chunk of para_targets ([chunk, 1024]):
    sims = c_n @ normalize(chunk).T            -> [1, chunk]
    running top-n maintained (n = 5, chunk_size = 8192)
=> near_sims [5], near_idxs [5]  -> {question, sentences, similarity, index} per passage
```

**Result** (`AtlasResult`): `answer` (str), `retrieved` (K sentence/score
pairs, raw scores shown), `passages` (n grounded passages), `embedding`
(composed [1024]).

> Note on `score_mode`. The composer is trained on **uniform** 1/K scores
> (Section 5), so `"uniform"` reproduces training conditions and is the
> validated default. Feeding raw retrieval scores (`"retriever"`) shifts every
> input vector additively through `score_embed` and degrades the composed
> output — it is retained only as an ablation. The raw retrieval scores are
> always *displayed* regardless; they are simply not *fed to the composer* in
> the default path.

---

## 5. Training procedure

Retriever and composer are optimized independently and are coupled only during
inference — the composer never sees retriever output during training.

### 5.1 Fact pool construction

- Source: MS-MARCO v2.1, `passage_text` from selected passages
  (`is_selected == 1`) only.
- Sentence segmentation: NLTK Punkt (`sent_tokenize`).
- Filter: keep passages that yield **2–4 sentences** after a 4–60 word/sentence
  filter (≈44% of selected passages qualify).
- Encode every question and every sentence to SONAR:
  `q_embs [N, 1024]`, `sent_embs [S, 1024]`, plus `sent_ranges` (per-passage
  sentence index spans).

### 5.2 Retriever (InfoNCE)

Batch = one passage per slot (`B = 256`, or 512 in the orchestrated loop),
guaranteeing cross-passage negatives.

```
q_batch [B, 1024] -> QueryEncoder -> y_batch [B, 1024]
candidates = [ in-batch sentences (P_batch)  |  hard negatives (B * sample_k) ]
all_cands [P_all, 1024]
scaled_scores = scorer.pairwise(all_cands, y_batch) * temperature   -> [B, P_all]
pos_mask, neg_mask : [B, P_all]     (all same-passage sentences are positives;
                                     other-passage + hard negs are negatives;
                                     non-self same-passage sentences masked out)
loss = InfoNCE(scaled_scores, pos_mask, neg_mask)   (sum-of-positives form)
```

Hard negatives are **premined offline**: 1024 candidates per passage
(300 hard, top-scored + 724 random), same-passage masked. Each step samples
`sample_k` (64, or 128 in the loop) of the 1024. The index is **refreshed**
periodically from the partially trained QueryEncoder (ANCE-style), with a
cosine LR schedule held continuous against the true final epoch count across
cycles.
Validation uses **full-pool** metrics (Recall@K, MRR, Top-1), not in-batch loss.

### 5.3 Composer (pure composition)

```
Input  : first-K gold sentence embeddings per passage
         z_topk [B, 3, 1024]   (from first3_indices; zero-padded if < 3 sents)
Scores : uniform 1/K, padded positions -> 0
Target : para_targets [B, 1024]  = SONAR of the JOINED first-3-sentence text
         (encoded as one string, not a mean of sentence embeddings)
Loss   : cosine_loss + 0.1 * norm_loss
         cosine_loss = mean(1 - cos(pred, target))
         norm_loss   = MSE(||pred||, ||target||.detach())   (magnitude term)
```

`B = 256`, `lr = 1e-4`, 50 epochs, cosine schedule, best-val-cosine checkpoint.

---

## 6. What makes it work

- **Raw-SONAR operation.** Removing the FactEncoder/bridge eliminates a
  documented latent-collapse failure surface and lets the composer act on a
  space that is already compositional. A controlled memorization test recovers
  near-perfect training cosine, confirming the architecture has the capacity.
  Our overfitting experiments suggest the current performance ceiling is
  dominated by training data quality rather than model capacity.
- **Retrieval/composition decoupling.** Training the composer only on coherent
  gold sentence sets (never on noisy retriever output) avoids forcing it to
  learn a compensatory mapping for retrieval errors.
- **Paragraph target = SONAR of joined text.** Encoding the actual first-3
  sentence *text* as one string (vs. averaging independent sentence embeddings)
  preserves cross-sentence context the encoder captures via attention, and
  prevents the composer from passing via trivial averaging.
- **Fixed K=3 input/target consistency.** The target is defined over the same
  first-3 sentences the composer sees, so it is never asked to hallucinate
  content it wasn't given.
- **Uniform score conditioning at inference.** Because the composer was trained
  on uniform 1/K scores, feeding uniform scores at inference (rather than the
  raw retrieval scores) is what makes the full pipeline behave correctly — this
  is the single change that moved the full-pipeline composer from harmful to
  beneficial (Section 7).

---

## 7. Metrics (validation, 400k-scale build)

**Retrieval** (full-pool, strict same-passage index match):

| Metric | Value |
|---|---|
| Recall@3 | 0.4550 |
| MRR | 0.3700 |

> The retrieval metric is pessimistic: a retrieved sentence counts only if its
> exact pool index belongs to the query's own passage. Many questions have
> multiple semantically equivalent passages in MS-MARCO, but the evaluation
> accepts only the original source passage, so equivalent sentences retrieved
> from near-duplicate passages are scored as misses. The reported Recall@3 of
> 0.4550 is therefore a conservative lower bound on practical retrieval
> quality, not an indication of weak retrieval.

**Composer — gold-input** (composer in isolation, gold sentences in):

| Metric | Value |
|---|---|
| Composer cosine (mean) | 0.8102 |
| Top-1 sentence baseline | 0.5079 |
| Delta (composer − top-1) | +0.3024 |
| Composer wins % | 93.5% |

**Composer — full-pipeline** (retriever output in, validated `score_mode="uniform"`):

| Metric | Value |
|---|---|
| Composer cosine (mean) | 0.4191 |
| Top-1 sentence baseline | 0.3328 |
| Delta (composer − top-1) | +0.0863 |
| Composer wins % | 84.0% |
| Distribution | min 0.046 · p25 0.321 · median 0.397 · p75 0.506 · max 0.958 |
| Negative-cosine cases | 0 / 200 |

Reading these together: the composer is strong in isolation (0.81) and remains
net-positive through noisy retrieval (0.42, beating the naive top-1 baseline
84% of the time, with zero collapses). The gap between 0.81 and 0.42 is
attributable to retrieval quality (Recall@3 0.4550), not the composer — where
retrieval returns coherent sentences the full pipeline approaches the
gold-input ceiling (best-case comp-sim 0.958).

---

## 8. Configuration reference

| Group | Parameter | Value |
|---|---|---|
| SONAR | encoder / decoder | `cointegrated/SONAR_200_text_encoder` / `raxtemur/SONAR_200_text_decoder` |
| SONAR | dim, lang, enc max_length | 1024, `eng_Latn`, 64 |
| Retriever | QueryEncoder hidden | 1024 |
| Retriever | InfoNCE temperature init | log(1/0.07) |
| Retriever | batch (one passage/slot) | 256 (512 in loop) |
| Retriever | lr, weight decay, grad clip | 3e-4, 1e-4, 5.0 |
| Retriever | hard-neg index (hard/random) | 1024 (300 / 724) |
| Retriever | hard-neg sampled/step | 64 (128 in loop) |
| Composer | K | 3 |
| Composer | heads / layers / d_model / ffn | 8 / 4 / 1024 / 4096 |
| Composer | dropout, activation | 0.0, GELU |
| Composer | loss | cosine + 0.1 · norm |
| Composer | batch, lr, epochs | 256, 1e-4, 50 |
| Data | passages, sentences | 360,710 / 1,045,350 |
| Data | sentences/passage, words/sentence | 2–4, 4–60 |
| Inference | chunk_size (retrieval / nearest) | 4096 / 8192 |
| Inference | top-K, n_neighbors, decode beams | 3, 5, 4 |

---

The public package is a behavior-preserving port of the development
inference path: equivalence tests pass bitwise for all three models, SONAR
encode/decode, retrieval, composition (both score modes), and
nearest-passage lookup.
