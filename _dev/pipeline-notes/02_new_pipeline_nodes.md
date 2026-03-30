# New Pipeline Nodes

Plan for the new matching pipeline nodes. Replaces the current matching stage
(`llm_summarise` -> `embed_ads` -> `embed_onet` -> `cosine_match` ->
`llm_filter_candidates`) with a three-stage flow that drops LLM summarisation
entirely.

## Current matching flow

```
sample_ads ─► llm_summarise ─► embed_ads ─► cosine_match ─► llm_filter_candidates
                                   ▲
prepare_onet_targets ─► embed_onet─┘
```

**Nodes removed:** `llm_summarise`, `embed_ads`, `embed_onet`, `cosine_match`
**Nodes kept:** `llm_filter_candidates` (modified to work with reranker output)
**Nodes added:** `embed_ads_v2`, `embed_onet_v2`, `cosine_candidates`, `rerank_candidates`

## New matching flow

```
sample_ads ──────────────────► embed_ads_v2 ──► cosine_candidates ──► rerank_candidates ──► llm_filter_candidates
                                    ▲                  ▲
prepare_onet_targets ──► embed_onet_v2 ────────────────┘
```

## New nodes

### `embed_ads_v2`

Embeds raw ad text directly (no LLM summarisation).

**Input:**
- `ad_ids: np.ndarray` (from `sample_ads`)

**Output:**
- `ad_ids: list[int]` (passthrough for downstream)

**What it does:**
1. Load ad texts from Adzuna DuckDB (title + description)
2. Read model config from `embed_models.toml` for the configured model
3. Apply `query_prefix` if the model has one (e.g. e5-large: `"query: "`)
4. Apply `prompt_name` if the model has one (e.g. Qwen3-Embed: `"query"`)
5. If `supports_prompt = true`, apply the task instruction from the node var
   `embed_task_prompt`
6. Call `aembed(texts, model=embedding_model, prompt=..., prompt_name=...)`
7. Save embeddings to `store/pipeline/{run_name}/embed_ads_v2/ad_embeddings.npy`

**Node vars:**
- `embedding_model` (global, inherited): model key from `embed_models.toml`
- `embed_task_prompt` (per-node): optional task instruction string, only used
  if the model has `supports_prompt = true`

**Key difference from current `embed_ads`:** No dependency on `llm_summarise`.
Takes `ad_ids` directly from `sample_ads`, loads raw text from the database,
and embeds it. The current `embed_ads` embeds LLM summaries from the
`llm_summarise` output.

**Storage:** `ad_embeddings.npy` (n_ads, embed_dim). Same format as current
`embed_ads` output.

---

### `embed_onet_v2`

Embeds O*NET occupations as single rich documents.

**Input:** None (source node, triggered by `prepare_onet_targets` signal)

**Output:**
- `out: bool` (signal that embeddings are ready)

**What it does:**
1. Load `onet_targets.parquet` (861 occupations)
2. Build a single rich text per occupation:
   `"{Title}\n\n{Description}\n\nKey tasks and skills: {Work Activities/Tasks/Skills}"`
3. Apply `document_prefix` or `document_prompt_name` if the model has one
4. Call `aembed(texts, model=embedding_model, ...)`
5. Save to `store/pipeline/{run_name}/embed_onet_v2/onet_embeddings.npy`
6. Also save the O*NET code order to `onet_codes.json` for index alignment

**Node vars:**
- `embedding_model` (global, inherited)

**Key difference from current `embed_onet`:** Produces a single embedding per
occupation (not two separate role/taskskill embeddings). The O*NET text is a
concatenation of all fields, letting the embedding model attend to whichever
parts are most relevant.

**Storage:** `onet_embeddings.npy` (861, embed_dim) + `onet_codes.json`

---

### `cosine_candidates`

Computes cosine similarity and selects top-N candidates per ad.

**Input:**
- `ad_ids: list[int]` (from `embed_ads_v2`)
- `onet_done: bool` (from `embed_onet_v2`, signal that O*NET embeddings exist)

**Output:**
- `ad_ids: list[int]` (passthrough)

**What it does:**
1. Load ad embeddings from `embed_ads_v2` output
2. Load O*NET embeddings from `embed_onet_v2` output
3. Compute cosine similarity matrix (n_ads x 861)
4. For each ad, select top-N candidates (N = `cosine_topk`, default 100)
5. Save candidates as parquet: columns `ad_id, rank, onet_code, cosine_score`

**Node vars:**
- `cosine_topk` (per-node): number of candidates to keep (default 100)
- `cosine_mode` (global, inherited): execution mode for cosine computation

**Key difference from current `cosine_match`:** Takes top-100 instead of top-10,
producing a wider candidate set for the reranker. Output format is simpler
(just ad_id, rank, code, score).

**Storage:** `candidates.parquet`

---

### `rerank_candidates`

Reranks cosine candidates using Qwen3-Reranker (generative reranker on GPU).

**Input:**
- `ad_ids: list[int]` (from `cosine_candidates`)

**Output:**
- `ad_ids: list[int]` (passthrough)

**What it does:**
1. Load cosine candidates from `cosine_candidates` output
2. Load raw ad texts from Adzuna DuckDB
3. Build O*NET document texts for the reranker (shorter than embedding texts:
   title + description, ~300 chars)
4. For each ad, score its top-N cosine candidates using the reranker
5. Call `arerank(queries, documents, top_k=rerank_topk, model=rerank_model)`
6. Save reranked results as parquet: `ad_id, rank, onet_code, rerank_score`

**Node vars:**
- `rerank_model` (per-node): model key from `rerank_models.toml`
  (default: `qwen3-reranker-8b-sbatch`)
- `rerank_topk` (per-node): number of top candidates to keep after reranking
  (default: 10)
- `sbatch_time` (per-node, inherited): Slurm walltime

**Processing strategy:** The reranker scores each ad against all its cosine
candidates. At 30M ads x 100 candidates, this is a large workload. Process
in batches (e.g. 10,000 ads per sbatch job) with resume support via
`ResultStore`.

Alternatively, if the reranker is fast enough, process all ads against all
861 occupations directly (bypassing the cosine filtering). This was tested
in notebook 06 and works, but may be too slow at 30M scale.

**Storage:** `reranked_matches.parquet`

---

### `llm_filter_candidates` (modified)

The existing LLM filter node, modified to read from `rerank_candidates`
output instead of `cosine_match` output.

**Changes needed:**
- Load candidates from `reranked_matches.parquet` instead of
  `matches.parquet`
- The candidates already have rerank scores, which could be included in the
  LLM prompt to help it prioritize
- Consider skipping the LLM call for ads where the reranker's top-1 score
  is well above the rest (high-confidence matches)

---

## Pipeline graph changes

### Nodes to add to `netrun.json`

```json
{"name": "embed_ads_v2", "factory": "netrun.node_factories.from_function", ...}
{"name": "embed_onet_v2", "factory": "netrun.node_factories.from_function", ...}
{"name": "cosine_candidates", "factory": "netrun.node_factories.from_function", ...}
{"name": "rerank_candidates", "factory": "netrun.node_factories.from_function", ...}
```

### Edges to add

```
sample_ads.ad_ids                          -> embed_ads_v2.ad_ids
embed_ads_v2.ad_ids                        -> cosine_candidates.ad_ids
embed_onet_v2.out                          -> cosine_candidates.onet_done
cosine_candidates.ad_ids                   -> rerank_candidates.ad_ids
rerank_candidates.ad_ids                   -> llm_filter_candidates.ad_ids
broadcast_onet_ready.out_0                 -> embed_onet_v2.__control_start_epoch__
```

### Edges to remove (old matching flow)

```
sample_ads.ad_ids                          -> llm_summarise.ad_ids
llm_summarise.successful_ad_ids            -> embed_ads.successful_ad_ids
embed_ads.ad_ids                           -> cosine_match.ad_ids
embed_onet.out                             -> cosine_match.onet_done
cosine_match.ad_ids                        -> llm_filter_candidates.ad_ids
broadcast_onet_ready.out_0                 -> embed_onet.__control_start_epoch__
```

### Nodes to remove (or disable)

- `llm_summarise`: no longer needed for matching
- `embed_ads`: replaced by `embed_ads_v2`
- `embed_onet`: replaced by `embed_onet_v2`
- `cosine_match`: replaced by `cosine_candidates`

Note: `llm_summarise` may still be useful for downstream feature extraction
(domain, level, seniority). If so, keep it but decouple from the matching
flow (run in parallel, triggered by `sample_ads` directly).

### New node vars to add

Global:
- `rerank_model` (string): model key for the reranker

Per-node (`embed_ads_v2`):
- `embed_task_prompt` (string): task instruction for instruction-following
  embedding models

Per-node (`cosine_candidates`):
- `cosine_topk` (int): number of candidates to keep (default 100)

Per-node (`rerank_candidates`):
- `rerank_model` (string, inherit from global): reranker model key
- `rerank_topk` (int): candidates to keep after reranking (default 10)
- `sbatch_time` (string, inherited): Slurm walltime

### `run_defs.toml` defaults

```toml
[defaults]
rerank_model = "qwen3-reranker-8b-sbatch"

[defaults.embed_ads_v2]
embed_task_prompt = "Instruct: Given a job advertisement, retrieve the occupational classification that best describes the type of work\nQuery: "

[defaults.cosine_candidates]
cosine_topk = 100

[defaults.rerank_candidates]
rerank_topk = 10
sbatch_time = "04:00:00"
```

## broadcast_onet_ready changes

The current `broadcast_onet_ready` has `num_outputs: 5` (one for each of:
`embed_onet`, `build_aspectt_vectors`, `score_presence`, `score_felten`,
`score_task_exposure`). Replacing `embed_onet` with `embed_onet_v2` means
`out_0` connects to `embed_onet_v2` instead.

No change to `num_outputs` needed, just rewire `out_0`.
