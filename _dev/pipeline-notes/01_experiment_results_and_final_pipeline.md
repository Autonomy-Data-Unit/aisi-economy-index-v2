# Final Pipeline Design: Experiment Results

Summary of experiments (notebooks 01-07 in `pts/scratch/experiments/`) and the
resulting pipeline design for job-ad-to-O*NET matching.

## What we tested

Seven experiment notebooks explored alternative matching approaches, evaluated
on 200 ads using GPT-5.4 as judge. Primary metric: "Top-10 has 4+" (fraction
of ads where at least one of the top-10 candidate occupations receives a
rating of 4 or 5 out of 5).

### Approaches that didn't work

**Centroid-residual matching** (notebook 02): Subtracting O*NET major group
centroids from embeddings before computing similarity. Intended to "zoom in"
on within-group differences. Result: dramatically worse than flat embedding
(52% vs 91.5% recall at top-5). Removing the group-level signal destroys
useful information rather than enhancing fine-grained discrimination.

**Classic cross-encoders** (notebooks 05, 06): Both bge-reranker-v2-m3 (278M)
and gte-reranker-modernbert-base (149M) performed worse than flat bi-encoder
embedding. bge-reranker scored 35.5% recall, gte-modernbert scored 86.5%,
both below the flat baseline of 95.5%. These models were trained on web search
passage retrieval (MS MARCO), which is too different from occupation
classification. The relevance signals they learned don't transfer.

**Hierarchical group filtering** (notebook 01): Classifying ads into O*NET
major groups first, then matching within groups. At N=3 groups, 90% coverage
with 86% candidate reduction. Viable for efficiency but provides no quality
improvement: it gives the same or worse results than flat matching, just faster.
Not worth the complexity.

**LLM reranking for recall** (notebooks 03, 04): GPT-5.4 reranking of embedding
candidates improves top-1 precision dramatically (3.67 to 4.58) but does not
improve recall (stays at 95.5%). The LLM is good at picking the best match from
a small pool but can't surface good candidates that the embedding missed.

### Approaches that worked

**Raw text embedding without LLM summarisation** (notebook 01): Embedding raw
ad text (title + description) with text-embedding-3-large achieves 61% good
match rate (rated 4+) on top-1, and 91.5% recall at top-5. This is without
any LLM involvement. The current pipeline's LLM summarisation step is not
needed for the embedding stage.

**Generative reranker: Qwen3-Reranker-8B** (notebook 06): The first open-source
reranker to beat flat embedding on our task. Achieves 98.5% recall at top-10
(vs 95.5% for flat embedding). Also improves top-1 quality (4.33 vs 3.67).
The key advantage over classic cross-encoders is instruction-following: we
defined a task-specific instruction ("Given a job advertisement, determine if
the occupation description accurately describes the type of work") that the
model uses to understand what "relevance" means in our context.

**Qwen3-Embedding-8B** (notebook 07): Open-source embedding model that
outperforms OpenAI text-embedding-3-large on every metric. Even without
reranking, Qwen3-Embedding-8B (97.5% recall) matches the API model with
reranker (97.5%). The two models only have 60.6% overlap in their top-100
candidates, meaning Qwen3 finds genuinely different and better candidates.

**Full open-source pipeline** (notebook 07): Qwen3-Embedding-8B +
Qwen3-Reranker-8B achieves 99.0% recall at top-10 with 4.34 mean top-1
rating. This is the best result across all experiments and runs entirely
on Isambard with no API dependencies.

## Results summary

| Pipeline | Mean top-1 | Top-10 has 4+ | Top-10 has 5 |
|----------|-----------|---------------|-------------|
| API flat (text-emb-3-large) | 3.67 | 95.5% | 81.0% |
| OSS flat (Qwen3-Embed-8B) | 3.80 | 97.5% | 85.5% |
| API + Qwen3-Reranker-8B | 4.30 | 97.5% | 85.0% |
| **OSS + Qwen3-Reranker-8B** | **4.34** | **99.0%** | **87.0%** |
| bge-reranker-v2-m3 (cross-enc) | 1.68 | 35.5% | 27.5% |
| gte-modernbert (cross-enc) | 2.38 | 86.5% | 65.5% |
| GPT-5.4 LLM reranking | 4.58 | 95.5% | 80.0% |

## Final pipeline design

```
Raw ad text ──► Qwen3-Embedding-8B ──► cosine top-100 ──► Qwen3-Reranker-8B ──► top-10 ──► LLM filter ──► top-1-3
                    (sbatch)              (sbatch)            (sbatch)                        (sbatch)
```

### Stage 1: Bi-encoder retrieval (Qwen3-Embedding-8B)

- Embed raw ad text (title + description, up to 32K tokens) directly
- No LLM summarisation needed
- Embed O*NET occupations as single rich documents (title + description +
  tasks + skills)
- Cosine similarity against all 861 occupations, take top-100
- Supports task-specific instructions via `prompt` parameter
  (`supports_prompt = true` in embed_models.toml)

### Stage 2: Generative reranker (Qwen3-Reranker-8B)

- Score each (ad, candidate occupation) pair using vLLM backend
- Uses yes/no logprob extraction with domain-specific instruction
- Takes top-100 from bi-encoder, produces top-10
- Runs on Isambard via sbatch (existing infrastructure)
- NOT a classic cross-encoder: generative rerankers with instruction-following
  dramatically outperform traditional cross-encoders on this task

### Stage 3: LLM filter (existing)

- One LLM call per ad on top-10 candidates
- Same role as current `llm_filter_candidates` node
- Could potentially be skipped for high-confidence reranker matches

### What changed vs the current pipeline

| Aspect | Current | New |
|--------|---------|-----|
| Ad representation | LLM summary (lossy, model-sensitive) | Raw text (lossless, deterministic) |
| Embedding model | bge-large-en-v1.5 (512 tokens) | Qwen3-Embedding-8B (32K tokens) |
| Retrieval depth | Top-10 | Top-100 |
| Reranking | None (cosine straight to LLM) | Qwen3-Reranker-8B (top-100 to top-10) |
| LLM calls per ad | 2 (summarise + filter) | 1 (filter only) |
| LLM on critical path | Yes (matching breaks without it) | Partially (reranker handles most work) |
| API dependencies | OpenAI embedding API | None (all on Isambard) |

### Cost comparison at 30M ads

**Current pipeline:**
- LLM summarise: 30M sbatch LLM calls (most expensive node)
- Embedding: 30M texts via API (~$7,800) or sbatch
- Cosine match: sbatch
- LLM filter: 30M sbatch LLM calls
- Total: 60M LLM calls + embedding cost

**New pipeline:**
- Qwen3-Embedding-8B: ~50-100 GPU-hours (single batch job, no LLM generation)
- Cosine top-100: ~10 GPU-hours
- Qwen3-Reranker-8B: needs calibration, estimate ~200-500 GPU-hours
  (30M ads x 100 candidates, generative inference but single-token output)
- LLM filter: 30M sbatch LLM calls (same as before)
- Total: 30M LLM calls + ~300-600 GPU-hours. No API cost.

The LLM summarisation step is eliminated entirely. The reranker adds GPU cost
but removes an LLM call per ad and produces better candidates for the filter.

## Infrastructure built

All infrastructure is in place to run the new pipeline:

- `llm_runner/rerank.pct.py`: `run_rerank()` with cross-encoder and vLLM backends
- `llm_runner/models.pct.py`: `load_rerank_model()` for cross-encoder loading
- `llm_runner/cli.pct.py`: `rerank` operation in CLI dispatcher
- `ai_index/utils/rerank.pct.py`: `rerank()`/`arerank()` with local/sbatch routing
- `config/rerank_models.toml`: model configs for cross-encoders and generative rerankers
- `llm_runner/embed.pct.py`: `prompt`/`prompt_name` parameters for instruction-
  following embedding models
- `config/embed_models.toml`: per-model prompt support metadata
  (`query_prefix`, `query_prompt_name`, `supports_prompt`)

## Next steps

1. **Integrate into netrun pipeline** as new nodes replacing `llm_summarise`,
   `embed_ads`, `cosine_match` with the new bi-encoder + reranker flow
2. **Calibrate GPU-hours** for Qwen3-Embedding-8B and Qwen3-Reranker-8B at
   30M-ad scale on Isambard
3. **Test with task-specific embedding instructions** (Qwen3-Embedding supports
   custom instructions but notebook 07 used the default prompt_name)
4. **Evaluate whether LLM filter can be skipped** for high-confidence reranker
   matches to further reduce LLM calls
5. **Consider Qwen3-Reranker-4B** as a faster alternative if 8B is too slow
   at scale (0.6B was tested and works, but quality vs 8B not yet compared)

## Key lessons learned

1. **Model choice matters more than architecture.** The same cross-encoder
   architecture (sentence-transformers CrossEncoder) fails with bge-reranker
   (35.5%) but the Voyage API reranker hit 99%. The difference is training
   data and model quality, not the approach.

2. **Instruction-following is crucial for domain-specific tasks.** Generative
   rerankers (Qwen3-Reranker) dramatically outperform traditional cross-encoders
   because they can be told what "relevance" means. This is why classic
   cross-encoders trained on MS MARCO fail: they learned web search relevance,
   not occupation classification relevance.

3. **Recall is more important than precision in the retrieval stage.** The
   downstream LLM filter handles precision. The embedding/reranker's job is
   to ensure the correct occupation is somewhere in the candidate set. This
   is why we evaluate "top-10 has 4+" rather than top-1 accuracy.

4. **Larger embedding models produce genuinely better candidates.** Qwen3-
   Embedding-8B and text-embedding-3-large have only 60.6% overlap in their
   top-100 candidates. The larger model doesn't just reorder, it finds
   different and better matches.

5. **The LLM summarisation step was the wrong approach.** It introduced
   model sensitivity (the original problem), lost information (83% of ads
   exceed the truncation point), and added cost. Raw text embedding is
   simpler, cheaper, and more accurate.
