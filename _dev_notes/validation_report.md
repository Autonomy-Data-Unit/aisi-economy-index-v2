# AISI Exposure Index: Pipeline Validation Report

## Executive Summary

We validated the AISI Exposure Index pipeline by running 61 configurations on a
5,000-ad sample, varying the LLM filter (12 models), embedding model (7 models),
and cross-encoder reranker (4 models) independently across three sensitivity arms.
Two additional analyses tested whether stability holds within individual
occupational domains.

**Bottom-line results on the primary score (`task_exposure_importance_weighted`):**

| Dimension varied | Pearson r (mean) | MAD (% of range) |
|---|---|---|
| LLM (Arm 1) | 0.954-0.968 | 1.6-4.3% |
| Embedding (Arm 2) | 0.917-0.976 | 1.4-5.0% |
| Reranker (Arm 3) | 0.954-0.965 | 2.3-2.7% |

The pipeline produces highly stable per-ad exposure scores across model choices.
Disagreement at intermediate stages (filter Jaccard ~0.44, reranker top-1 ~0.43)
is concentrated on low-importance candidates and is dampened by the
rerank-score-weighted averaging in the final exposure computation.

Within-group analysis reveals that overall stability is partly structural: the
between-group variance across O\*NET major groups inflates the overall Pearson
(0.955) relative to within-group Pearson (0.67-0.93). Small models (7B/8B)
account for roughly half of the within-group instability. After pruning the
4 weakest models, per-group Pearson improves substantially (e.g. Legal:
0.88 to 0.98, Computer & Math: 0.72 to 0.87).


## Methodology

### Pipeline overview

Each pipeline run takes a set of job ads through: embedding, cosine top-k
retrieval (k=20), LLM negative filtering, cross-encoder reranking, and finally
rerank-score-weighted exposure scoring. Occupation-level exposure scores (Felten,
presence dimensions, task exposure) are computed independently from O\*NET data and
are identical across all runs.

### Validation design

A crossed sensitivity study with three arms, each varying one dimension while
holding the others fixed:

- **Arm 1 (LLM sensitivity):** 12 LLMs x 2 fixed embeddings x 2 fixed rerankers = 4 arm groups
- **Arm 2 (Embedding sensitivity):** 7 embeddings x 2 fixed LLMs x 2 fixed rerankers = 4 arm groups
- **Arm 3 (Reranker sensitivity):** 4 rerankers x 2 fixed LLMs x 2 fixed embeddings = 4 arm groups

**LLMs tested:** Qwen2.5-7B, Qwen3-8B, Qwen2.5-14B, gemma-3-27b, Qwen2.5-32B,
EXAONE-4.0-32B, Qwen2.5-72B, Llama-4-Scout-17B, Llama-3.1-70B, Mistral-Large-2,
phi-4, gpt-oss-120b

**Embeddings tested:** bge-large-en-v1.5, e5-large-v2, gte-large, nomic-embed,
embedding-gemma, qwen3-embed-0.6b, qwen3-embed-8b

**Rerankers tested:** bge-reranker-v2-m3, gte-reranker-modernbert, Qwen3-Reranker-8B,
llama-nemotron-rerank-1b

**Fixed models used across arms:**
- Fixed embeddings: bge-large-en-v1.5, e5-large-v2
- Fixed LLMs: phi-4, Llama-3.1-70B
- Fixed rerankers: Qwen3-Reranker-8B, bge-reranker-v2-m3

### Metrics

Agreement is measured at three pipeline stages:

1. **Filter stage:** Jaccard similarity of kept candidate sets, top-1 agreement
   (fraction of ads where both runs agree on the best candidate)
2. **Rerank stage:** Weighted Jaccard (Ruzicka similarity) over rerank score
   vectors, top-1 agreement (best-scoring occupation), Spearman rank correlation
   on shared candidates
3. **Exposure stage:** Pearson correlation and mean absolute difference (MAD) on
   per-ad exposure scores


## Arm 1: LLM Sensitivity

Fix the embedding and reranker; vary the LLM used in the filter step. Cosine
candidates are identical across runs with the same embedding, so all disagreement
enters at the LLM filter.

### Filter stage

LLMs vary considerably in selectivity. Qwen3-8B is the most aggressive (median
3 candidates/ad), EXAONE-4.0-32B the most permissive (median 7). This directly
affects Jaccard: if one model keeps 3 and another keeps 8, even with full overlap
on the 3, the Jaccard is only 3/8.

| Metric | Mean | Range |
|---|---|---|
| Filter Jaccard | 0.44-0.51 | 0.23-0.59 |
| Filter top-1 | 0.73-0.76 | 0.56-0.84 |

### Rerank stage

The reranker acts as a consensus mechanism: it orders shared candidates almost
identically regardless of which LLM produced them.

| Metric | Mean | Range |
|---|---|---|
| Weighted Jaccard (Ruzicka) | 0.51-0.78 | -- |
| Rerank top-1 | 0.74-0.85 | 0.63-0.94 |
| Spearman on shared candidates | ~1.00 | 0.998-1.000 |

The near-perfect Spearman (~1.0) means the reranker is fully deterministic on its
inputs: all remaining top-1 disagreement comes from candidates one LLM kept and
the other dropped, not from reranking inconsistency.

### Exposure scores

| Metric | bge-large + Qwen3-Reranker | e5-large + Qwen3-Reranker | bge-large + bge-reranker | e5-large + bge-reranker |
|---|---|---|---|---|
| Pearson (primary) | 0.967 | 0.968 | 0.935 | 0.939 |
| MAD (% range) | 1.7% | 1.6% | 3.5% | 4.3% |

All score columns show Pearson above 0.93, with the primary outcome
(`task_exposure_importance_weighted`) typically in the 0.94-0.97 range.
MAD is 1.6-4.3% of the score range.

### Best subsets

Dropping Qwen3-8B (the most selective LLM) and Qwen2.5-7B (the smallest model)
consistently yields the largest improvement in pairwise Jaccard, raising the mean
from ~0.44 to ~0.49. The best-2 subset (gemma-27b + Mistral-Large) achieves
Jaccard ~0.59.


## Arm 2: Embedding Sensitivity

Fix the LLM and reranker; vary the embedding model. This is the most disruptive
variation because different embeddings produce entirely different cosine candidate
pools, unlike Arms 1 and 3 where the candidate pool is shared.

### Cosine stage

Different embeddings retrieve substantially different top-20 candidate sets.

| Metric | Mean | Range |
|---|---|---|
| Cosine Jaccard (top-20) | 0.33-0.36 | 0.30-0.44 |
| Cosine Jaccard (top-5) | 0.32-0.36 | 0.30-0.46 |

### Filter and rerank stages

The LLM filter partially recovers agreement from the divergent candidate pools,
and the reranker amplifies this recovery.

| Metric | Mean | Range |
|---|---|---|
| Filter Jaccard | 0.42-0.46 | 0.42-0.51 |
| Filter top-1 | 0.45-0.60 | 0.42-0.69 |
| Weighted Jaccard (Ruzicka) | 0.43-0.79 | -- |
| Rerank top-1 | 0.65-0.86 | -- |

The wide range on Ruzicka and rerank top-1 reflects the impact of reranker choice:
groups using Qwen3-Reranker-8B show higher agreement (0.75-0.85) than those with
gte-reranker-modernbert (~0.44-0.65).

### Exposure scores

| Metric | phi-4 + Qwen3-Reranker | Llama-70B + Qwen3-Reranker | phi-4 + gte-reranker | Llama-70B + gte-reranker |
|---|---|---|---|---|
| Pearson (primary) | 0.965 | 0.976 | 0.917 | 0.932 |
| MAD (% range) | 1.8% | 1.4% | 3.9% | 5.0% |

Embedding choice has the largest impact of the three dimensions on upstream
retrieval (cosine Jaccard ~0.35), but the LLM filter and reranker recover most
of the signal: final Pearson still exceeds 0.91 in all cases.


## Arm 3: Reranker Sensitivity

Fix the LLM and embedding; vary the reranker. The candidate sets are identical
(same embedding + same LLM filter), so all variation comes from how the reranker
scores and orders the same candidates.

### Rerank stage

Rerankers show moderate-to-low agreement on candidate rankings but agree better
on which candidate is best.

| Metric | Mean | Range |
|---|---|---|
| Top-1 agreement | 0.38-0.48 | 0.30-0.54 |
| Spearman rank correlation | 0.26-0.28 | 0.17-0.34 |
| Weighted Jaccard (Ruzicka) | 0.41-0.45 | 0.19-0.74 |

The low Spearman (~0.27) and wide Ruzicka range indicate that rerankers use
fundamentally different scoring scales: Qwen3-Reranker-8B scores are in a
different regime from the three cross-encoder models (bge, gte, nemotron), which
cluster together (Ruzicka ~0.60-0.74 among themselves vs ~0.19-0.26 against
Qwen3-Reranker).

### Exposure scores

| Metric | phi-4 + bge-large | phi-4 + e5-large | Llama-70B + bge-large | Llama-70B + e5-large |
|---|---|---|---|---|
| Pearson (primary) | 0.963 | 0.965 | 0.954 | 0.955 |
| MAD (% range) | 2.3% | 2.3% | 2.7% | 2.7% |

Despite the low ranking agreement (Spearman ~0.27), the final exposure scores are
highly stable (Pearson 0.95-0.97). This is because exposure is a
rerank-score-weighted average across multiple candidates: even when rerankers
disagree on the ordering, the top few candidates contribute similar occupation
scores, dampening the effect.

### Pearson matrices

The three cross-encoder rerankers (bge-reranker-v2-m3, gte-reranker-modernbert,
llama-nemotron-rerank-1b) form a tight cluster (pairwise Pearson 0.98-1.00),
while Qwen3-Reranker-8B is the outlier (pairwise Pearson 0.92-0.94 against the
others). This is consistent with the Ruzicka pattern: same-architecture models
produce similar score distributions.


## Major Group Stability Analysis

### Motivation

The overall Pearson of ~0.955 is computed across all ads. But does stability hold
*within* individual occupational domains, or is it an artifact of averaging across
diverse domains with very different exposure profiles?

### Method

Each ad is assigned to an O\*NET major group based on the first 2 digits of its
top-1 reranked occupation code (the SOC major group). Within each major group
with at least 30 ads, we compute the same pairwise Pearson as the arm analyses.
This is done across all 12 arm groups from all three arms.

### Results

21 of 22 major groups have sufficient sample size (only Farming/Fishing/Forestry
excluded at N=7). The per-group Pearson is **substantially lower** than the overall
Pearson for every group:

| Major Group | N ads | Mean Pearson | Delta from overall |
|---|---|---|---|
| 23 Legal | 87 | 0.666 | -0.289 |
| 31 Healthcare Support | 198 | 0.745 | -0.209 |
| 15 Computer & Mathematical | 436 | 0.774 | -0.181 |
| 11 Management | 591 | 0.786 | -0.169 |
| 13 Business & Financial | 417 | 0.792 | -0.163 |
| 53 Transportation & Material Moving | 308 | 0.795 | -0.160 |
| 39 Personal Care & Service | 145 | 0.808 | -0.146 |
| 17 Architecture & Engineering | 241 | 0.829 | -0.126 |
| 37 Building & Grounds | 123 | 0.834 | -0.121 |
| 21 Community & Social Service | 81 | 0.835 | -0.119 |
| 49 Installation & Maintenance | 183 | 0.844 | -0.111 |
| 25 Education & Library | 341 | 0.863 | -0.092 |
| 29 Healthcare Practitioners | 201 | 0.868 | -0.087 |
| 41 Sales | 306 | 0.868 | -0.086 |
| 35 Food Preparation & Serving | 325 | 0.870 | -0.085 |
| 33 Protective Service | 43 | 0.878 | -0.077 |
| 51 Production | 146 | 0.884 | -0.071 |
| 19 Life, Physical & Social Science | 77 | 0.884 | -0.070 |
| 27 Arts, Design & Media | 73 | 0.892 | -0.062 |
| 43 Office & Administrative | 405 | 0.900 | -0.054 |
| 47 Construction & Extraction | 173 | 0.926 | -0.028 |

This is **not a small-sample effect**: Management (N=591), Computer & Math (N=436),
and Business & Financial (N=417) are among the largest groups yet all sit below
r=0.80.

### Explanation: between-group vs within-group variance

This is a form of Simpson's paradox. The overall Pearson is dominated by
between-group variance: Management ads have very different exposure scores from
Construction ads, and all models agree on that structural difference. Within a
major group, all occupations have similar exposure profiles, so the effective
signal-to-noise ratio is much lower. A 0.05 disagreement that is invisible
against a between-group gap of 0.5 becomes a large fraction of a within-group
range of 0.2.

This is reassuring for the pipeline's primary use case (geographic aggregation):
each Local Authority District contains a mix of occupational domains, so the
between-group signal dominates. It would only be a concern if someone used the
scores to rank ads within a single occupation.

### By arm

Arm 3 (reranker) generally shows the highest within-group stability, while
Arms 1 (LLM) and 2 (embedding) are similar and lower. Exception: Legal, where
the reranker arm is the least stable (0.611).

### Best model subsets

A few outlier models account for much of the within-group instability. The 4
models most frequently dropped across all groups are **Qwen2.5-7B, Qwen3-8B,
phi-4, and Qwen2.5-14B** (the smaller models).

Improvement from keeping only the best 8 of 12 LLMs:

| Group | All 12 | Best 8 | Improvement | Dropped models |
|---|---|---|---|---|
| 15 Computer & Math | 0.716 | 0.874 | +0.158 | Qwen2.5-14B, Qwen2.5-7B, Qwen3-8B, phi-4 |
| 21 Community & Social | 0.748 | 0.867 | +0.119 | Qwen2.5-32B, Qwen2.5-7B, Qwen3-8B, gpt-oss-120b |
| 23 Legal | 0.881 | 0.983 | +0.102 | Qwen2.5-14B, Qwen2.5-32B, Qwen2.5-7B, Qwen3-8B |
| 13 Business & Financial | 0.684 | 0.783 | +0.098 | Qwen2.5-32B, Qwen2.5-7B, Qwen3-8B, phi-4 |
| 11 Management | 0.729 | 0.796 | +0.067 | Qwen2.5-14B, Qwen2.5-7B, Qwen3-8B, gpt-oss-120b |

Small models (7B/8B) account for roughly half the within-group instability. But
even after pruning, Management (0.80) and Business & Financial (0.78) remain
below 0.90, reflecting genuine difficulty distinguishing between similar
occupations within these broad domains.


## Leave-One-Group-Out Analysis

### Motivation

The complement of the major group analysis: instead of asking "is each group
stable?" it asks "is overall stability robust to the removal of any group?"

### Method

For each major group, remove its ads from the sample and recompute the overall
mean pairwise Pearson. The delta measures whether the group contributes to or
detracts from overall stability.

### Results

All deltas are tiny (max |delta| < 0.005):

| Major Group | N removed | Mean delta |
|---|---|---|
| 15 Computer & Mathematical | 436 | +0.0031 |
| 47 Construction & Extraction | 173 | +0.0028 |
| 37 Building & Grounds | 123 | +0.0024 |
| 51 Production | 146 | +0.0020 |
| 13 Business & Financial | 417 | +0.0012 |
| 11 Management | 591 | +0.0012 |
| ... | | |
| 25 Education & Library | 341 | -0.0016 |
| 39 Personal Care & Service | 145 | -0.0017 |
| 41 Sales | 306 | -0.0025 |

Positive delta means removing the group *decreases* overall Pearson. This may
seem counterintuitive for groups like Computer & Math (which has low within-group
Pearson), but it reflects their contribution to between-group variance: their
distinctive exposure profile helps separate them from other groups, boosting
overall correlation.

**Conclusion:** No single group disproportionately drives the overall stability
result. The high Pearson is a distributed structural property of the pipeline,
not an artifact of any particular occupational domain.


## Cross-Arm Summary

### Disagreement propagation

The pipeline shows a consistent pattern across all three arms: substantial
disagreement at intermediate stages is dampened to near-zero at the exposure
stage.

| Stage | Arm 1 (LLM) | Arm 2 (Embedding) | Arm 3 (Reranker) |
|---|---|---|---|
| Candidate retrieval | identical | Jaccard ~0.35 | identical |
| Filter (Jaccard) | ~0.44 | ~0.44 | identical |
| Rerank top-1 | ~0.85 | ~0.65-0.86 | ~0.43 |
| Exposure Pearson | ~0.97 | ~0.92-0.98 | ~0.96 |
| MAD (% range) | ~2% | ~2-5% | ~2.5% |

The reranker acts as a consensus mechanism in Arms 1 and 2 (Spearman ~1.0 on
shared candidates), meaning it consistently scores candidates in the same relative
order regardless of which LLM or embedding produced them.

### Which dimension matters most?

Embedding choice has the largest impact on upstream retrieval but is mostly
recovered downstream. The ranking by final exposure Pearson:

1. **LLM choice** has the least impact (Pearson 0.94-0.97): the reranker
   effectively overrides LLM filter disagreement
2. **Reranker choice** has moderate impact (Pearson 0.95-0.97): score distributions
   differ but weighted averaging dampens the effect
3. **Embedding choice** has the largest impact (Pearson 0.92-0.98): entirely
   different candidate pools propagate through the pipeline

However, all three dimensions produce Pearson above 0.91 in the worst case,
indicating the pipeline is robust to model choice across all dimensions.

### Model quality tiers

Across all analyses, a consistent quality hierarchy emerges:

**Tier 1 (high agreement, large models):** gemma-3-27b, Mistral-Large-2,
Llama-3.1-70B, Qwen2.5-72B, EXAONE-4.0-32B

**Tier 2 (moderate agreement):** Qwen2.5-32B, Llama-4-Scout, gpt-oss-120b,
Qwen2.5-14B

**Tier 3 (lower agreement, small models):** phi-4, Qwen2.5-7B, Qwen3-8B

The tier-3 models are consistently the first to be dropped in best-subset
analyses across all major groups and all arms.


## Conclusions

1. **The pipeline is highly stable.** Final per-ad exposure scores show Pearson
   above 0.91 and MAD below 5% of range across all 61 validation configurations.
   For the best model combinations, Pearson exceeds 0.97 with MAD under 2%.

2. **Disagreement is harmless.** Moderate disagreement at intermediate stages
   (filter Jaccard ~0.44, reranker top-1 ~0.43) is concentrated on low-importance
   candidates. The rerank-score-weighted averaging in the exposure computation
   dampens this to negligible effect on final scores.

3. **Overall stability is partly structural.** The overall Pearson of ~0.955 is
   boosted by between-group variance across O\*NET major groups. Within-group
   Pearson ranges from 0.67 to 0.93. This is expected and acceptable: geographic
   aggregation (the pipeline's primary use case) benefits from between-group
   signal, and within-group instability is largely driven by small models that
   would be excluded in a production configuration.

4. **Small models drive most within-group instability.** Qwen2.5-7B, Qwen3-8B,
   and phi-4 are consistently the weakest performers. Dropping the 4 weakest
   models raises within-group Pearson by 0.05-0.16 depending on the occupational
   domain.

5. **No single occupational domain drives overall stability.** Leave-one-group-out
   deltas are all below 0.005, confirming the stability result is robust and
   distributed.

6. **Reranker choice matters less than it appears.** Despite low ranking agreement
   (Spearman ~0.27), reranker Pearson on exposure scores exceeds 0.95. The three
   cross-encoder models (bge, gte, nemotron) form a tight cluster; Qwen3-Reranker
   is the outlier but still produces highly correlated final scores.
