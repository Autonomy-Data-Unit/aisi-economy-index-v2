# Reranker Architecture Analysis

## Context

The initial Arm 3 validation (4 rerankers) showed Qwen3-Reranker-8B as an
"outlier" against 3 cross-encoders. We expanded to 8 rerankers to test whether
this was a generative-vs-cross-encoder split:

| Model | Params | Type | Scoring |
|---|---|---|---|
| bge-reranker-v2-m3 | 0.6B | Cross-encoder | Logit |
| gte-reranker-modernbert | 0.2B | Cross-encoder | Logit |
| nemotron-rerank-1b | 1B | Cross-encoder | Logit |
| lb-reranker-0.5B | 0.5B | Generative (Qwen2.5) | 1-7 scale expectation |
| Qwen3-Reranker-0.6B | 0.6B | Generative (Qwen3) | yes/no logits |
| Qwen3-Reranker-4B | 4B | Generative (Qwen3) | yes/no logits |
| Qwen3-Reranker-8B | 8B | Generative (Qwen3) | yes/no logits |
| rank1-3b | 3B | Generative (Qwen2.5) | Reasoning + true/false logits |


## Key finding: the split is not generative vs cross-encoder

Three clusters emerge, but they do **not** align with architecture:

**Cluster 1 -- Cross-encoders + lb-reranker** (Ruzicka 0.79, Pearson 0.98-1.00):
bge, gte, nemotron, and lb-reranker-0.5B. Despite lb-reranker being a generative
model (Qwen2.5-0.5B, 1-7 Likert scoring), it clusters tightly with the
cross-encoders on both weight distribution (Ruzicka 0.75-0.85) and final exposure
scores (Pearson 0.985-0.997).

**Cluster 2 -- Qwen3-Reranker family** (Ruzicka 0.56, Pearson 0.93-0.98):
0.6B, 4B, and 8B. The 4B-8B pair is tight (Ruzicka 0.68, Pearson 0.975).
The 0.6B is looser (Ruzicka 0.49-0.50, Pearson 0.93-0.94), likely limited
by model capacity.

**Cluster 3 -- rank1** (Pearson 0.89-0.94 against everything):
The reasoning reranker is the most distant from all other models. Highest
agreement is with Qwen3-4B/8B (Pearson 0.93-0.94), lowest with gte-reranker
(Pearson 0.89).

The real axis of differentiation is the **scoring mechanism**, not the
architecture:
- Direct relevance scoring (cross-encoders and lb-reranker's 1-7 scale) cluster
  together
- Binary yes/no probability (Qwen3 family) forms a second cluster
- Reasoning-then-scoring (rank1) forms a third


## Rank correlation is universally low

Within-cluster Spearman is only ~0.30 for both the CE/lb cluster and the Qwen3
family. This means **no reranker family agrees on candidate rankings**, even
among architecturally similar models.

The one exception is Qwen3-4B/8B (Spearman 0.60), the only pair with
meaningfully correlated rankings. They share the same training data and
approach; the size difference is the only variable.

| Cluster | Spearman | Ruzicka | Pearson (exposure) |
|---|---|---|---|
| CE + lb-reranker (within) | 0.30 | 0.79 | 0.98-1.00 |
| Qwen3 family (within) | 0.31 | 0.56 | 0.93-0.98 |
| CE/lb vs Qwen3 (cross) | 0.28 | 0.43 | 0.91-0.95 |
| All vs rank1 | 0.11-0.52 | 0.23-0.54 | 0.89-0.94 |


## Why high Ruzicka but low Spearman in the CE cluster?

The CE cluster has Ruzicka 0.79 (models assign similar normalized weights to
candidates) but Spearman only 0.30 (the rank orderings are quite different).
This can happen when models assign similar magnitude scores but in a different
order -- the weight distributions overlap substantially (high Ruzicka) even
though the rankings diverge (low Spearman). Since exposure is computed as a
weighted average, similar weight distributions produce similar final scores
regardless of ranking order.


## Qwen3-Reranker produces highest cross-LLM and cross-embedding stability

Despite being the "outlier" in direct reranker comparison, Qwen3-Reranker-8B
produces the most stable final exposure scores when other dimensions vary:

| Fixed reranker | Arm 1 Pearson (LLM stability) | Arm 2 Pearson (Embedding stability) |
|---|---|---|
| Qwen3-Reranker-8B | 0.967-0.968 | 0.965-0.976 |
| bge-reranker-v2-m3 | 0.935-0.939 | 0.917-0.932 |

This suggests Qwen3-Reranker is a stronger consensus mechanism that better
overrides upstream disagreement, even though its score distributions differ
from the cross-encoder family.


## Exposure score stability across all 8 rerankers

The pipeline remains robust. Mean pairwise Pearson on the primary score
(`task_exposure_importance_weighted`) across all arm groups:

| Arm group | Mean Pearson | MAD (% range) |
|---|---|---|
| phi-4 + bge-large | 0.948 | 2.9% |
| phi-4 + e5-large | 0.950 | 2.9% |
| Llama-70B + bge-large | 0.935 | 3.3% |
| Llama-70B + e5-large | 0.938 | 3.3% |

Even the worst pairwise Pearson (rank1 vs gte-reranker at 0.89) is acceptable.
The pipeline's weighted averaging dampens ranking disagreement into negligible
exposure score differences.


## Practical implications

1. **Model selection does not critically matter for the final index.** All 8
   rerankers produce exposure scores with Pearson > 0.89 against each other.

2. **Qwen3-Reranker-8B remains the recommended production choice.** It produces
   the highest cross-model stability in Arms 1 and 2, despite disagreeing with
   cross-encoders on internal rankings.

3. **The "outlier" framing was misleading.** The original 4-model analysis
   suggested a generative-vs-CE split. With 8 models, the picture is more
   nuanced: lb-reranker (generative) clusters with CEs, while rank1 (also
   generative) is the true outlier.

4. **rank1's reasoning approach adds cost without improving stability.** It took
   4-5 hours per Slurm job (vs minutes for other rerankers) and produces the
   most divergent scores. The reasoning chain doesn't converge to consensus
   with other approaches.

5. **Qwen3-0.6B is too small for reliable reranking.** Its Spearman with its
   own family (0.14-0.19) suggests it lacks the capacity to produce consistent
   rankings. The 4B model is the minimum viable size for the Qwen3 reranker
   family.
