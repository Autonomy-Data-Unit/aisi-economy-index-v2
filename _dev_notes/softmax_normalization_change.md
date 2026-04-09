# Softmax Normalization Change: Before/After Comparison

## What changed

The `compute_job_ad_exposure` node previously used L1 normalization to convert
rerank scores to per-ad weights:

```python
# Before: L1 norm
weight = rerank_score / sum(rerank_scores_for_ad)
```

This was replaced with min-max scaling followed by softmax(T=1):

```python
# After: softmax on min-max scaled scores
scaled = (score - min) / (max - min)   # per-ad [0, 1]
weight = exp(scaled) / sum(exp(scaled))  # softmax
```

## Why

Different reranker architectures produce fundamentally different score
distributions:

- **Cross-encoders** (bge, gte, nemotron) and **lb-reranker**: relatively flat
  scores, top-1 candidate gets 31-42% of weight after L1 norm
- **Qwen3-Reranker family**: peaked scores (most candidates near zero, one
  high), top-1 gets 69-73% of weight
- **rank1**: extremely peaked, top-1 gets 79% (median 92%) of weight

Under L1 norm, "the same normalization" produces very different effective
behaviors: nearly-uniform averaging for CEs vs nearly-top-1 selection for
Qwen3/rank1. Softmax on min-max scaled scores produces more consistent weight
distributions across architectures.


## Impact on Arm 3 (reranker sensitivity)

The primary motivation. Pearson on `task_exposure_importance_weighted`:

| Arm group | Before (L1) | After (softmax) | Change |
|---|---|---|---|
| phi-4 + bge-large | 0.948 | **0.986** | +0.038 |
| phi-4 + e5-large | 0.950 | **0.987** | +0.037 |
| Llama-70B + bge-large | 0.935 | **0.990** | +0.055 |
| Llama-70B + e5-large | 0.938 | **0.991** | +0.053 |

MAD on primary score:

| Arm group | Before | After | Change |
|---|---|---|---|
| phi-4 + bge-large | 2.3% | **1.3%** | -1.0pp |
| phi-4 + e5-large | 2.3% | **1.4%** | -0.9pp |
| Llama-70B + bge-large | 2.7% | **1.5%** | -1.2pp |
| Llama-70B + e5-large | 2.7% | **1.5%** | -1.2pp |

The cluster structure in the Pearson matrix essentially disappeared. Before,
the min pairwise Pearson was 0.912 (rank1 vs gte-reranker). After, it's 0.982.
All 8 rerankers now agree at r > 0.98.


## Impact on Arm 1 (LLM sensitivity)

Slight degradation, as expected. The softmax flattens weights, so disagreement
on lower-ranked candidates (which LLMs disagree on more) now contributes more
to the final score.

| Arm group | Before (L1) | After (softmax) | Change |
|---|---|---|---|
| bge-large + Qwen3-Reranker | 0.967 | 0.944 | -0.023 |
| bge-large + bge-reranker | 0.935 | 0.928 | -0.007 |
| e5-large + Qwen3-Reranker | 0.968 | 0.946 | -0.022 |
| e5-large + bge-reranker | 0.939 | 0.937 | -0.002 |

The degradation is concentrated in the Qwen3-Reranker groups (-0.022), because
Qwen3's peaked distribution was previously acting as a consensus mechanism
(concentrating weight on the one candidate all LLMs agree on). With softmax,
more candidates contribute, exposing LLM disagreement on the lower-ranked ones.

The bge-reranker groups barely change (-0.002 to -0.007) because the CE
distribution was already relatively flat.

MAD on primary score (Arm 1):

| Arm group | Before | After |
|---|---|---|
| bge-large + Qwen3-Reranker | 1.7% | 3.3% |
| bge-large + bge-reranker | 3.5% | 3.7% |
| e5-large + Qwen3-Reranker | 1.6% | 3.2% |
| e5-large + bge-reranker | 4.3% | 3.9% |

Qwen3-Reranker groups see MAD increase (from 1.7% to 3.3%), while bge-reranker
groups are roughly unchanged. The values converge: all groups now have MAD in the
3.2-3.9% range, whereas before they ranged from 1.6% to 4.3%.


## Impact on major group stability

| Metric | Before | After |
|---|---|---|
| Overall mean Pearson | 0.955 | 0.957 |
| Best group (Construction) | 0.926 | 0.923 |
| Worst group (Legal) | 0.666 | 0.597 |

The overall Pearson is essentially unchanged. Per-group Pearson is similar for
most groups but slightly worse for the already-weak groups (Legal dropped from
0.666 to 0.597). This is the same mechanism as Arm 1: flatter weights expose
more LLM/embedding disagreement on lower-ranked candidates.


## Net assessment

The softmax normalization is a clear net positive:

1. **Arm 3 (reranker) improved dramatically**: Pearson 0.94 to 0.99, MAD 2.5% to
   1.4%. The reranker architecture clusters disappeared entirely.

2. **Arm 1 (LLM) degraded slightly**: Pearson dropped by ~0.02, MAD increased by
   ~1.5pp for Qwen3-Reranker groups. But the absolute levels remain acceptable
   (Pearson > 0.93, MAD < 4%).

3. **Arm 1 groups converged**: The gap between Qwen3-Reranker groups and
   bge-reranker groups narrowed. Before, Qwen3-Reranker appeared to produce
   much better LLM stability (Pearson 0.97 vs 0.94), but this was an artifact
   of its peaked weighting. With softmax, both rerankers produce similar LLM
   stability (0.93-0.95).

4. **The trade-off is asymmetric**: Arm 3 gained 0.04-0.05 Pearson while Arm 1
   lost 0.02. And the Arm 1 loss is partly illusory (it was inflated by the
   peaked weighting acting as a consensus filter).

5. **The normalization makes cross-reranker comparisons fair**: With L1 norm,
   comparing a CE and Qwen3-Reranker was comparing "mean of candidates" vs
   "top-1 selection" -- fundamentally different operations. With softmax, all
   rerankers use similar weight distributions, so Pearson comparisons are
   apples-to-apples.
