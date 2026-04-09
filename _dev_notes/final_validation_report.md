# AISI Exposure Index: Final Validation Report

## 1. Overview

This report documents the full validation of the AISI Exposure Index pipeline,
covering model sensitivity, occupational-domain stability, reranker architecture
analysis, and score normalization. The validation was conducted on a 5,000-ad
sample with 92 pipeline configurations spanning 12 LLMs, 7 embedding models,
and 8 rerankers.

**Pipeline summary:** Job ads are embedded, matched to O\*NET occupations via
cosine top-k retrieval (k=20), filtered by an LLM (negative selection), scored
by a cross-encoder or generative reranker, and then converted to per-ad
exposure scores via a rerank-score-weighted average of occupation-level metrics.

**Primary outcome metric:** `task_exposure_importance_weighted` (O\*NET task-level
AI exposure, importance-weighted, aggregated to occupation level via the scoring
nodes, then mapped to ads via the reranker weights).

**Score normalization:** Rerank scores are converted to per-ad weights using
softmax on min-max scaled scores with temperature T=0.7. This was selected after
systematic analysis of the trade-off between cross-reranker agreement and
within-group stability (see Section 6).


## 2. Validation Design

A crossed sensitivity study with three arms, each varying one pipeline dimension:

| Arm | Varied | Fixed | Groups | Runs per group |
|---|---|---|---|---|
| 1 (LLM) | 12 LLMs | 2 embeddings x 2 rerankers | 4 | 7-12 |
| 2 (Embedding) | 7 embeddings | 2 LLMs x 2 rerankers | 4 | 2-7 |
| 3 (Reranker) | 8 rerankers | 2 LLMs x 2 embeddings | 4 | 8 |

**LLMs:** Qwen2.5-7B/14B/32B/72B, Qwen3-8B, gemma-3-27b, Mistral-Large-2,
Llama-3.1-70B, Llama-4-Scout, EXAONE-4.0-32B, phi-4, gpt-oss-120b

**Embeddings:** bge-large, e5-large, gte-large, nomic-embed, embedding-gemma,
qwen3-embed-0.6b, qwen3-embed-8b

**Rerankers (8 models, 3 architectures):**
- Cross-encoders: bge-reranker-v2-m3, gte-reranker-modernbert, nemotron-rerank-1b
- Generative (Qwen3 yes/no): Qwen3-Reranker-0.6B, 4B, 8B
- Generative (other): lb-reranker-0.5B (1-7 Likert), rank1-3b (reasoning + true/false)

**Benchmark:** GPT-5.4 + text-embedding-3-large + voyage-rerank-2.5 (frontier
API models).


## 3. Arm Results

### 3.1 Arm 1: LLM Sensitivity

Fix embedding + reranker, vary LLM. Cosine candidates are shared; all
disagreement enters at the LLM filter.

**Pearson on primary score:**

| Arm group | Mean Pearson |
|---|---|
| bge-large + Qwen3-Reranker-8B | 0.951 |
| bge-large + bge-reranker-v2-m3 | 0.928 |
| e5-large + Qwen3-Reranker-8B | 0.953 |
| e5-large + bge-reranker-v2-m3 | 0.936 |

MAD: 2.9-3.7% of range. The reranker acts as a consensus mechanism: Spearman
on shared candidates is ~1.0, meaning all remaining exposure disagreement comes
from which candidates the LLM filter kept, not from how the reranker scored
them.

### 3.2 Arm 2: Embedding Sensitivity

Fix LLM + reranker, vary embedding. The most disruptive variation: different
embeddings produce entirely different cosine candidate pools (Jaccard ~0.35).

**Pearson on primary score:** 0.928-0.966 depending on arm group. MAD: 2.0-4.5%.

Despite the lowest upstream agreement, the LLM filter and reranker recover most
of the signal: Pearson exceeds 0.92 in all cases.

### 3.3 Arm 3: Reranker Sensitivity

Fix LLM + embedding, vary reranker. All 8 rerankers score the same candidate
set.

**Pearson on primary score:**

| Arm group | Mean Pearson |
|---|---|
| phi-4 + bge-large | 0.974 |
| phi-4 + e5-large | 0.975 |
| Llama-70B + bge-large | 0.979 |
| Llama-70B + e5-large | 0.981 |

MAD: 1.8-2.1%. The softmax T=0.7 normalization produces consistent weight
distributions across reranker architectures. The min pairwise Pearson is 0.970
(vs 0.912 under the original L1 normalization).

**Pairwise Pearson matrix (phi-4 + bge-large, primary score):**

All 28 reranker pairs achieve Pearson > 0.96. The previous cluster structure
(cross-encoders vs generative) is largely dissolved. Qwen3-4B and 8B remain the
tightest pair (0.994), while the weakest pairs involve Qwen3-0.6B or rank1-3b
(~0.97).


## 4. Benchmark Comparison

Each validation run is compared against the frontier API benchmark. Agreement
with the benchmark by reranker:

| Reranker | Exposure Pearson | Rerank Top-1 | Rerank Spearman |
|---|---|---|---|
| Qwen3-Reranker-8B | **0.941** | **0.673** | 0.614 |
| Qwen3-Reranker-4B | **0.940** | **0.675** | **0.626** |
| rank1-3b | 0.930 | 0.581 | 0.482 |
| lb-reranker-0.5B | 0.917 | 0.583 | 0.584 |
| bge-reranker-v2-m3 | 0.914 | 0.448 | 0.361 |
| gte-reranker-modernbert | 0.911 | 0.456 | 0.359 |
| Qwen3-Reranker-0.6B | 0.911 | 0.468 | 0.275 |
| nemotron-rerank-1b | 0.906 | 0.382 | 0.289 |

The generative rerankers (Qwen3-4B/8B) have the highest benchmark agreement.
The benchmark itself uses voyage-rerank-2.5 (a generative API reranker),
consistent with generative rerankers as a class producing more similar scoring
behaviour to frontier models.

Agreement by LLM: gpt-oss-120b leads (0.953), small models (Qwen3-8B, Qwen2.5-7B)
trail (0.907-0.908). By embedding: Qwen3-Embedding-8B leads (0.941), bge-large
trails (0.919).


## 5. Major Group Stability

### 5.1 Stratified analysis

Each ad is assigned to an O\*NET major group (first 2 digits of the SOC code).
Within each group, the same pairwise Pearson is computed. 21 of 22 groups have
sufficient sample size (N >= 30).

**Overall mean Pearson:** 0.956

**Per-group Pearson (primary score, averaged across all 12 arm groups):**

| Group | N ads | Mean Pearson |
|---|---|---|
| 23 Legal | 87 | 0.607 |
| 31 Healthcare Support | 198 | 0.736 |
| 15 Computer & Mathematical | 436 | 0.767 |
| 11 Management | 591 | 0.787 |
| 13 Business & Financial | 417 | 0.800 |
| 53 Transportation | 308 | 0.802 |
| ... (15 more groups) | | 0.81-0.92 |
| 47 Construction & Extraction | 173 | 0.922 |

Within-group Pearson is lower than the overall 0.956 for every group. This is a
form of Simpson's paradox: the overall Pearson is boosted by between-group
variance (Management ads have very different exposure from Construction ads, and
all models agree on that structural difference). Within a group, the effective
signal-to-noise ratio is lower because all occupations have similar exposure
profiles.

This is acceptable for the pipeline's primary use case (geographic aggregation),
where each Local Authority District contains a mix of occupational domains and
the between-group signal dominates.

### 5.2 Best model subsets

The weakest groups are partly driven by small LLMs. Dropping the 4 worst models
(Qwen2.5-7B, Qwen3-8B, gpt-oss-120b, and one of Qwen2.5-14B/phi-4) raises
per-group Pearson by 0.03-0.13 depending on the group. A production
configuration using only tier-1 LLMs would have better within-group stability.

### 5.3 Leave-one-group-out

All LOO deltas are below 0.005. No single occupational domain
disproportionately drives the overall stability. The high Pearson is a
distributed structural property.


## 6. Reranker Architecture Analysis

### 6.1 Score distributions

The original L1 normalization (`weight = score / sum`) produced very different
effective behaviours depending on the reranker:

| Model type | Top-1 weight (mean) | Effective behaviour |
|---|---|---|
| Cross-encoders + lb-reranker | 0.31-0.42 | Near-uniform averaging |
| Qwen3-Reranker family | 0.69-0.73 | Top-1 dominated |
| rank1-3b | 0.79 (median 0.92) | Near-argmax |

### 6.2 Normalization analysis

We tested multiple normalization schemes on the trade-off between cross-reranker
agreement (Arm 3 Pearson) and cross-LLM agreement (Arm 1 Pearson):

| Scheme | Arm 3 Pearson | Arm 1 Pearson | Qwen3 Ruzicka |
|---|---|---|---|
| L1 norm (original) | 0.948 | 0.967 | 0.556 |
| softmax T=0.50 | 0.955 | 0.957 | 0.665 |
| **softmax T=0.70** | **0.974** | **0.951** | **0.721** |
| softmax T=1.00 | 0.986 | 0.944 | 0.790 |
| rank-based (Borda) | 0.985 | -- | -- |
| uniform | 1.000 | -- | -- |

The uniform result (Pearson 1.000) confirms that all rerankers share the same
candidate set and the candidates have similar mean exposure scores. The
disagreement is entirely about how to weight them, not which candidates are
relevant.

**T=0.7 was selected** as the best balance: Qwen3-family Ruzicka exceeds 0.70,
Arm 3 Pearson reaches 0.974 (up from 0.948), and Arm 1 Pearson remains at 0.951
(a modest 0.016 drop that reflects genuine LLM disagreement previously masked by
peaked weighting).

### 6.3 Cluster structure

With 8 rerankers and softmax T=0.7, the original 3-cluster pattern (CE vs Qwen3
vs rank1) is largely dissolved in the exposure Pearson. The Ruzicka still shows
some structure (CE/lb cluster at 0.69, Qwen3 at 0.72, cross-cluster at 0.67)
but the gap between within-cluster and cross-cluster Ruzicka has narrowed from
0.36 (under L1 norm) to 0.04.

Spearman rank correlation remains low (~0.30) regardless of normalization. The
rerankers genuinely disagree on candidate orderings. But this doesn't matter for
the final scores because the candidate occupations within each ad have similar
exposure profiles.


## 7. Model Quality Tiers

Across all analyses, a consistent hierarchy emerges:

**LLMs (by benchmark agreement and cross-model stability):**
- Tier 1: gpt-oss-120b, gemma-3-27b, Mistral-Large-2, Qwen2.5-72B, Qwen2.5-32B
- Tier 2: EXAONE-4.0-32B, Llama-3.1-70B, Llama-4-Scout, phi-4, Qwen2.5-14B
- Tier 3: Qwen2.5-7B, Qwen3-8B (consistently first to be dropped in best-subset analysis)

**Rerankers (by benchmark Pearson):**
- Tier 1: Qwen3-Reranker-8B (0.941), Qwen3-Reranker-4B (0.940)
- Tier 2: rank1-3b (0.930), lb-reranker-0.5B (0.917)
- Tier 3: bge-reranker-v2-m3 (0.914), gte-reranker-modernbert (0.911),
  Qwen3-Reranker-0.6B (0.911), nemotron-rerank-1b (0.906)

**Embeddings (by benchmark Pearson):**
- Tier 1: Qwen3-Embedding-8B (0.941), EmbeddingGemma (0.940)
- Tier 2: Qwen3-Embedding-0.6B (0.933), gte-large (0.931), e5-large (0.927)
- Tier 3: nomic-embed (0.919), bge-large (0.919)


## 8. Conclusions

1. **The pipeline is robust to model choice.** All 92 configurations produce
   per-ad exposure scores with Pearson > 0.89 against each other on the
   primary score. The best configurations exceed 0.97. MAD is typically
   2-4% of the score range.

2. **Reranker architecture does not meaningfully affect the final index.** With
   softmax T=0.7 normalization, all 8 rerankers (3 cross-encoders, 3 Qwen3
   generative, 1 Likert-scale, 1 reasoning) agree at Pearson > 0.97. The
   original apparent "generative vs cross-encoder" split was primarily a score
   calibration artifact, not a fundamental disagreement about which occupations
   match which ads.

3. **Overall stability is partly structural.** The between-group variance across
   O\*NET major groups contributes significantly to the overall Pearson of 0.956.
   Within-group Pearson ranges from 0.61 (Legal) to 0.92 (Construction). This
   is expected and acceptable for geographic aggregation, where each LAD
   contains a mix of occupational domains.

4. **Small models are the main source of instability.** Qwen2.5-7B, Qwen3-8B,
   and Qwen3-Reranker-0.6B consistently underperform. A production configuration
   excluding these would have higher within-group stability.

5. **Qwen3-Reranker-4B/8B are the recommended production rerankers.** They have
   the highest benchmark agreement (0.94) and the highest cross-model stability.
   The 4B model slightly outperforms the 8B on English MTEB-R benchmarks and
   costs half the compute.

6. **The softmax T=0.7 normalization is a principled improvement** over L1 norm.
   It produces consistent weight distributions across reranker architectures,
   eliminates the cluster structure in cross-reranker comparisons, and makes
   the pipeline's use of multiple candidate matches genuine rather than
   effectively top-1 selection.


## 9. Validation Notebooks

| Notebook | Purpose |
|---|---|
| `analyse_job_ads_pipeline_arm1` | LLM sensitivity (filter Jaccard, rerank metrics, exposure Pearson) |
| `analyse_job_ads_pipeline_arm2` | Embedding sensitivity (cosine Jaccard, propagation through stages) |
| `analyse_job_ads_pipeline_arm3` | Reranker sensitivity (8 models, architecture comparison) |
| `analyse_benchmark_comparison` | Agreement with frontier API benchmark |
| `analyse_major_group_stability` | Per-O\*NET-major-group Pearson with best-subset analysis |
| `analyse_leave_one_group_out` | LOO analysis confirming no group drives overall stability |
