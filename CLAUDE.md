# CLAUDE.md

## Project Overview

**ai-index** (AISI Economy Index v2) is a productionized data pipeline for analyzing AI exposure in the economy. It matches job advertisements to O\*NET occupations and computes AI impact metrics (ASPECTT vectors, AI exposure scores, seniority/job zone).

This is a clean rewrite of the old repository at `/Users/lukas/dev/20260208_e22t36__aisi-economy-index`, which was a collection of manually-run notebooks. The v2 uses **netrun** for orchestrating the data pipeline and **nblite** for literate programming development.

## Pipeline DAG

The pipeline is defined in `config/netrun.json`. It currently contains 20 nodes (17 function nodes + 2 broadcast + 1 join) and 23 edges. The pipeline has three main stages: job ad processing (blue), O\*NET exposure scoring (green), and index construction (red).

```
  ┌─── Job Ad Processing ───────────────────────────────────────────────────────┐
  │                                                                             │
  │  fetch_adzuna ──► sample_ads ──► llm_summarise ──► embed_ads ──┐            │
  │                                                                 │            │
  │  fetch_onet ──► prepare_onet_targets ──► broadcast_onet_ready ──┤            │
  │                                          (1→5)  │ │ │          │            │
  │                                                  │ │ │     embed_onet       │
  │                                                  │ │ │          │            │
  │                                                  │ │ │     cosine_match      │
  │                                                  │ │ │          │            │
  │                                                  │ │ │     llm_filter ──► broadcast_filter_done
  │                                                  │ │ │                      (1→2)
  └──────────────────────────────────────────────────┼─┼─┼──────────────────────┘
                                                     │ │ │
  ┌─── O*NET Exposure Scoring ───────────────────────┼─┼─┼──────────────────────┐
  │                                                  │ │ │                       │
  │  build_aspectt_vectors ◄─────────────────────────┘ │ │                       │
  │  score_presence ◄──────────────────────────────────┘ │                       │
  │  score_felten ◄──────────────────────────────────────┤                       │
  │  score_task_exposure ◄───────────────────────────────┘                       │
  │       │            │           │                                             │
  │       └──► join_scores ◄───────┘                                             │
  │                │                                                             │
  │         combine_onet_exposure                                                │
  └──────────────────────────────────────────────────────────────────────────────┘
                        │
  ┌─── Index Construction ──────────────────────────────────────────────────────┐
  │                     │                                                       │
  │  compute_job_ad_exposure ◄── broadcast_filter_done                          │
  │         │                                                                   │
  │    aggregate_geo                                                            │
  │                                                                             │
  │  compute_job_ad_aspectt_vectors ◄── broadcast_filter_done (currently disabled)│
  └─────────────────────────────────────────────────────────────────────────────┘
```

### Nodes (20 total: 17 function + 2 broadcast + 1 join)

**Data ingestion & preparation:**
- `fetch_onet` (run_on_startup) — Download and extract O\*NET 30.0 database to `store/inputs/onet/`. No output ports; signals `epoch_finished`.
- `fetch_adzuna` (run_on_startup) — Download raw Adzuna job ads from S3 to DuckDB, deduplicate. Signals `epoch_finished` to trigger `sample_ads`. Node vars: `fetch_years`.
- `sample_ads` — Sample job ads for processing (or pass through all if `sample_n=0`). Output: `ad_ids`.
- `prepare_onet_targets` — Filter O\*NET occupations (remove 33 public-sector-only) and build text descriptions for embedding. Reads O\*NET tables from disk, writes `store/inputs/onet_targets.parquet`. Triggered by `fetch_onet` `epoch_finished` signal via `start_epoch` control, signals `epoch_finished`. Node vars: `onet_exclude_public_sector` (bool), `onet_top_n` (int).

**Job ad processing (matching):**
- `llm_summarise` — Run LLM to extract structured summaries from job ads using structured JSON output (`json_schema` parameter). Processes ads in configurable chunks with incremental DuckDB writes and resume support. Input: `ad_ids`. Output: `successful_ad_ids` (list[int]). Prompts loaded from `prompt_library/` via `system_prompt` and `user_prompt` node vars. Node vars: `summarise_resume` (bool), `summarise_max_retries` (int), `system_prompt`, `user_prompt`.
- `embed_ads` — Build text descriptions from LLM summaries (`[domain] short_description` + tasks/skills) and embed with configured model in chunks. Input: `successful_ad_ids`. Output: `ad_ids` (list[int]). Stores embeddings as BLOBs in DuckDB via ResultStore (supports resume). Node vars: `embed_chunk_size` (int).
- `embed_onet` — Embed O\*NET occupation text descriptions (role + tasks/skills). Reads `onet_targets.parquet`. Output: `out` (bool). Writes `.npy` files to `store/pipeline/{run_name}/embed_onet/`. Triggered via `start_epoch` control from `broadcast_onet_ready`, signals `epoch_finished`.
- `cosine_match` — Weighted dual cosine similarity between ad and O\*NET embeddings. Inputs: `ad_ids` (list[int] from embed_ads), `onet_done` (bool from embed_onet). Output: `ad_ids` (list[int]). Loads `.npy` embeddings from disk, computes top-K role and taskskill cosine scores, combines with `combined = alpha * role + (1-alpha) * task`. Writes `matches.parquet` to `store/pipeline/{run_name}/cosine_match/`. Node vars: `cosine_alpha` (float), `cosine_chunk_size` (int).
- `llm_filter_candidates` — LLM negative selection to filter cosine match candidates. For each ad, builds a prompt with job context (title, sector, domain, tasks, raw description excerpt) and candidate occupations. LLM identifies which candidates to DROP, keeping 2-3 functional matches. Input: `ad_ids` (list[int] from cosine_match). Output: `ad_ids` (list[int]). Uses `run_batched` with `ResultStore` for incremental DuckDB writes and resume support. Writes `filtered_matches.parquet` to `store/pipeline/{run_name}/llm_filter_candidates/`. Node vars: `filter_resume` (bool), `filter_max_retries` (int), `system_prompt`, `user_prompt`.

**O\*NET exposure scoring:**
- `build_aspectt_vectors` — Build ASPECTT numeric vectors (Abilities, Skills, Knowledge, Work Activities) from O\*NET database tables. Reads `onet_targets.parquet` for filtered occupation codes, loads 4 O\*NET tables, pivots each by Level (LV) and Importance (IM) scales into ~157-dimensional feature vectors per occupation. Writes `aspectt_vectors.npz` to `store/inputs/aspectt_vectors/`. Triggered via `start_epoch` control from `broadcast_onet_ready`. Runs in parallel with the ad processing branch and score nodes.
- `score_presence` — Compute humanness/presence scores per O\*NET occupation across three dimensions: physical, emotional, and creative. Each dimension is defined by curated O\*NET element IDs (Work Context, Work Activities, Skills). Normalized to [0, 1] and averaged. Output: `out` (pd.DataFrame with `onet_code`, `presence_physical`, `presence_emotional`, `presence_creative`, `presence_composite`). Writes to `store/outputs/onet_exposure_scores/score_presence/`. Triggered via `start_epoch` control from `broadcast_onet_ready`.
- `score_felten` — Compute Felten AIOE (AI Occupational Exposure) scores per O\*NET occupation using the ability-application relatedness methodology from Felten et al. (2021). Uses a progress-weighted average of ability-application relatedness scores across 10 AI applications. Output: `out` (pd.DataFrame with `onet_code`, `felten_score`). Writes to `store/outputs/onet_exposure_scores/score_felten/`. Node vars: `felten_alpha` (float), `felten_scenario` (str). Triggered via `start_epoch` control from `broadcast_onet_ready`.
- `score_task_exposure` — Classify each O\*NET task statement via LLM into a 3-level AI exposure scale (0=no change, 1=human+LLM collaboration, 2=LLM independent), then aggregate to occupation level. Uses structured JSON output (`TaskExposureModel`). Output: `out` (pd.DataFrame with `onet_code`, `task_exposure_mean`, `task_exposure_importance_weighted`). Writes to `store/outputs/onet_exposure_scores/score_task_exposure/{llm_model}/`. Node vars: `llm_model` (inherited global), `system_prompt`, `user_prompt`. Triggered via `start_epoch` control from `broadcast_onet_ready`.
- `combine_onet_exposure` — Merge all O\*NET score DataFrames into a single combined exposure table. Receives a dict of `{name: pd.DataFrame}` from `join_scores`. Joins on `onet_code` and saves `scores.csv` to `store/outputs/onet_exposure_scores/`. Output: `out` (pd.DataFrame). No node vars.

**Index construction:**
- `compute_job_ad_exposure` — Map occupation-level exposure scores to individual job ads using filtered match weights. Column-agnostic — computes weighted averages for whatever score columns exist in the combined exposure table. Processes in chunks. Inputs: `ad_ids` (list[int] from `broadcast_filter_done`), `exposure_scores` (pd.DataFrame from `combine_onet_exposure`). Output: `ad_ids` (list[int]). Writes `ad_exposure.parquet` to `store/pipeline/{run_name}/compute_job_ad_exposure/`. Node vars: `exposure_chunk_size` (int).
- `aggregate_geo` — Aggregate ad-level AI exposure scores by Local Authority District (LAD22CD). Joins `ad_exposure.parquet` with the Adzuna ads table via DuckDB, computes per-LAD mean scores. Outputs `geo_lad.csv` to `store/outputs/{run_name}/`. Input: `ad_ids` (list[int] from `compute_job_ad_exposure`). No node vars beyond `run_name`.
- `compute_job_ad_aspectt_vectors` (currently **disabled**) — Compute per-ad weighted ASPECTT vectors from filtered occupation matches. Uses `ResultStore` for resume support. Inputs: `ad_ids` (from `broadcast_filter_done`), `aspectt_done` (bool from `build_aspectt_vectors`). Node vars: `aspectt_chunk_size` (int).

**Infrastructure nodes:**
- `broadcast_onet_ready` — Broadcast node (`num_outputs: 5`). Fans out `prepare_onet_targets` `epoch_finished` signal to `embed_onet`, `build_aspectt_vectors`, `score_presence`, `score_felten`, and `score_task_exposure`.
- `broadcast_filter_done` — Broadcast node (`num_outputs: 2`). Fans out `llm_filter_candidates` `ad_ids` to `compute_job_ad_aspectt_vectors` and `compute_job_ad_exposure`.
- `join_scores` — Join node (synchronization barrier). Waits for packets on ports `presence`, `felten`, and `task_exposure` from the three score nodes, then emits a single dict on `out` to `combine_onet_exposure`.

### Node Storage Convention

Three storage locations, each serving a different purpose:

**`store/inputs/`** — Run-independent source data (shared across all runs):
```
store/inputs/
├── onet/                    # O*NET database (fetch_onet)
├── onet_targets.parquet     # Filtered O*NET occupations (prepare_onet_targets)
├── adzuna.duckdb            # Adzuna ads (fetch_adzuna)
├── aspectt_vectors/         # .npz — ASPECTT feature vectors (build_aspectt_vectors)
├── lad22_lookup.csv         # ONS LAD22 name lookup table
└── AIOE_DataAppendix.xlsx   # Felten et al. ability-application matrix
```

**`store/pipeline/{run_name}/`** — Run-specific pipeline intermediates:
```
store/pipeline/{run_name}/
├── llm_summarise/               # DuckDB (ResultStore) — incremental LLM results
├── embed_ads/                   # DuckDB (ResultStore) — embeddings as BLOBs
├── embed_onet/                  # .npy — dense embedding arrays
├── cosine_match/                # .parquet — tabular match results
├── llm_filter_candidates/       # DuckDB (ResultStore) + .parquet — filtered matches
├── compute_job_ad_exposure/     # .parquet — per-ad exposure scores
└── compute_job_ad_aspectt_vectors/  # DuckDB (ResultStore) — per-ad ASPECTT vectors
```

**`store/outputs/`** — Final outputs:
```
store/outputs/
├── onet_exposure_scores/        # O*NET occupation-level scores (run-independent)
│   ├── score_presence/          # .csv — humanness/presence scores
│   ├── score_felten/            # .csv — Felten AIOE scores
│   ├── score_task_exposure/     # .csv + .parquet — per-model task exposure
│   │   └── {llm_model}/        # Subdirectory per LLM model
│   └── scores.csv              # Combined exposure table (combine_onet_exposure)
└── {run_name}/                  # Run-specific final outputs
    └── geo_lad.csv              # LAD-level geographic aggregation (aggregate_geo)
```

Storage format guidelines:
- **DuckDB** (via `ResultStore`): Nodes with incremental/transactional writes and resume/retry (LLM processing nodes)
- **Parquet**: Final tabular outputs read downstream (match results, filtered matches, ad exposure)
- **NumPy** (`.npy`/`.npz`): Dense numeric arrays (embeddings, feature vectors)
- **CSV**: Small human-readable outputs (O\*NET scores, geographic aggregations)

### Node Function Paths

Each node is a module at `ai_index.nodes.<name>` (developed as `pts/ai_index/nodes/<name>.pct.py`).

### Old Pipeline Reference

The old pipeline (now fully rebuilt in v2) had these stages:
1. **Embedding Generation** — `nbs/isambard/2026_01/00_transformers_for_origin_and_target.ipynb`
2. **Cosine Similarity Search** — `nbs/isambard/2026_01/01_cosine_sim_target_vs_origin.ipynb`
3. **LLM Filtering** — `nbs/isambard/2026_01/02_llm_negative_selection.ipynb`
4. **Impact Computation** — `nbs/helpers/AI_impact_occupation_and_seniority_job_zone.ipynb`
- **O\*NET fetch & build**: `nbs/helpers/fetch_and_build_onet.ipynb`
- **Exposure scoring**: `nbs/__scratch/exposure_score_pipeline/`

## Tech Stack

- **netrun** - Flow-based data pipeline orchestration (nodes, edges, packets, epochs)
- **nblite** (>=1.1.12) - Notebook-driven literate programming (`.pct.py` -> `.ipynb` -> Python modules)
- **Python 3.12+**
- **uv** - Package management
- **sentence-transformers** - BGE-large embeddings
- **torch** - GPU inference
- **pandas / polars** - Data manipulation
- **pydantic** - Configuration and data validation
- **isambard_utils** - Isambard HPC interaction (SSH, rsync, Slurm, env setup)
- **llm_runner** - Local/remote GPU model execution (embeddings, LLM generate, cosine similarity)
- **adulib[llm]** - LLM API abstraction (used in api execution mode)

## Running the Pipeline

The pipeline is run via `run_pipeline_async(run_name)` (or the `run-pipeline` CLI entry point). The flow:

1. Load `.env` via `dotenv`
2. Load `run_defs.toml` — `_load_run_defs()` parses the TOML file
3. Resolve run definition — `_resolve_run_defs(run_defs, run_name)` merges `[defaults]` with `[runs.<run_name>]`, producing `(global_node_vars, per_node_vars)` dicts. Scalar values become global node vars; subtable dicts become per-node overrides.
4. Load netrun config — `NetConfig.from_file(netrun.json, global_node_vars=..., node_vars=...)` injects the resolved values into the graph's unfilled `NodeVariable` placeholders
5. Execute — `async with Net(config) as net:` starts the net, then loops `run_until_blocked()` until no progress

Run name is determined by: explicit argument > `RUN_NAME` env var > `"baseline"`.

### Key files
- `pts/ai_index/run_pipeline.pct.py` — `run_pipeline_async()`, `_load_run_defs()`, `_resolve_run_defs()`
- `pts/ai_index/const.pct.py` — Path constants (`store_path`, `inputs_path`, `outputs_path`, `onet_exposure_scores_path`, `aspectt_vectors_path`, config paths)
- `config/run_defs.toml` — Run definitions
- `config/netrun.json` — Pipeline graph with unfilled node_var placeholders

## Configuration Files

### `run_defs.toml` — Run definitions

Defines named pipeline configurations. `[defaults]` provides base values; `[runs.<name>]` overrides them. Scalar values become global node_vars; subtable dicts (e.g. `[defaults.llm_summarise]`) become per-node overrides. All values are accessible via `ctx.vars` in nodes.

**Convention:** Default values for all node variables (both global and per-node) live in `run_defs.toml`, not in `netrun.json`. The `netrun.json` only declares variable names and types as unfilled placeholders.

```toml
[defaults]
sample_n = 10            # N = sample N ads, 0 = full run
sample_seed = 42
embedding_model = "text-embedding-3-large"   # Key into embed_models.toml
cosine_mode = "api"      # "api", "local", or "sbatch"
topk = 10                # Top-K candidates for cosine similarity
llm_model = "gpt-5.2"   # Key into llm_models.toml
llm_batch_size = 1000    # Number of prompts per LLM call
llm_max_new_tokens = 220 # Max tokens per LLM response
llm_max_concurrent_batches = 1   # Max concurrent batch LLM calls

[defaults.fetch_adzuna]
fetch_years = "all"

[defaults.prepare_onet_targets]
onet_exclude_public_sector = true
onet_top_n = 10

[defaults.embed_ads]
embed_chunk_size = 50000

[defaults.cosine_match]
cosine_alpha = 0.4               # Role score weight (task weight = 1 - alpha)
cosine_chunk_size = 50000

[defaults.llm_summarise]
summarise_resume = true          # Resume from previous partial run
summarise_max_retries = 0        # Retry rounds for failed ads
system_prompt = "llm_summarise/main/system"  # Path in prompt_library/
user_prompt = "llm_summarise/main/user"      # Path in prompt_library/

[defaults.llm_filter_candidates]
filter_resume = true             # Resume from previous partial run
filter_max_retries = 0           # Retry rounds for failed ads
system_prompt = "llm_filter/main/system"     # Path in prompt_library/
user_prompt = "llm_filter/main/user"         # Path in prompt_library/

[defaults.compute_job_ad_aspectt_vectors]
aspectt_chunk_size = 50000

[defaults.compute_job_ad_exposure]
exposure_chunk_size = 50000

[defaults.score_felten]
felten_alpha = 0.5
felten_scenario = "baseline_2025"

[defaults.score_task_exposure]
llm_model = "gpt-5.2"                                # Override of inherited global
system_prompt = "score_task_exposure/main/system"
user_prompt = "score_task_exposure/main/user"

[runs.baseline]          # Inherits all defaults
[runs.test]              # Quick test (10 ads, otherwise defaults)
sample_n = 10
[runs.test_api]          # API mode test (10 ads, text-embedding-3-large, gpt-5.2)
[runs.test_local]        # Local CPU test (10 ads, bge-large-mac, qwen-0.5b-mac)
[runs.test_sbatch]       # Isambard sbatch test (10 ads, bge-large-sbatch, qwen-7b-sbatch)
```

### `embed_models.toml` — Embedding model configs

Model-key-based lookup. Callers pass a model key (e.g., `embed(texts, model="bge-large-local")`), and the config resolves the execution mode and parameters.

```toml
[defaults.api]           # Defaults for mode="api"
batch_size = 200
[defaults.local]         # Defaults for mode="local"
device = "cuda"
dtype = "float16"
batch_size = 64
[defaults.sbatch]        # Defaults for mode="sbatch"
dtype = "float16"
batch_size = 64
gpus = 1
job_name = "embed"
time = "01:00:00"
setup = true

[models."text-embedding-3-small"]
mode = "api"
model = "text-embedding-3-small"
[models."text-embedding-3-large"]
mode = "api"
model = "text-embedding-3-large"
[models.bge-large-local]
mode = "local"
model = "BAAI/bge-large-en-v1.5"
[models.bge-large-mac]    # CPU/float32 for Mac testing
mode = "local"
model = "BAAI/bge-large-en-v1.5"
device = "cpu"
dtype = "float32"
[models.gemini-embed]     # Gemini API with retry config
mode = "api"
model = "gemini/gemini-embedding-001"
batch_size = 10
retry_delay = 65
max_retries = 20
[models.bge-large-sbatch]
mode = "sbatch"
model = "BAAI/bge-large-en-v1.5"
```

Resolution: `_load_model_config(config_path, model_key)` looks up `models.<key>`, reads `mode`, merges `defaults.<mode>` with the model entry, returns `(mode, merged_dict)`.

### `llm_models.toml` — LLM model configs

Same structure as `embed_models.toml`. Model keys map to execution modes and model parameters.

```toml
[defaults.api]           # (empty — no special defaults)
[defaults.local]
max_new_tokens = 60
device = "cuda"
dtype = "float16"
backend = "transformers"
batch_size = 128
[defaults.sbatch]
max_new_tokens = 60
dtype = "fp8"
backend = "vllm"
gpus = 1
job_name = "llm_generate"
time = "02:00:00"
setup = true

[models."gpt-5.2"]
mode = "api"
model = "openai/gpt-5.2"
[models.gemini-2-flash]
mode = "api"
model = "gemini/gemini-2.0-flash"
[models.qwen-7b-local]
mode = "local"
model = "Qwen/Qwen2.5-7B-Instruct"
[models."qwen-0.5b-mac"]  # CPU/float32 for Mac testing
mode = "local"
model = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cpu"
dtype = "float32"
batch_size = 16
[models.qwen-7b-sbatch]
mode = "sbatch"
model = "Qwen/Qwen2.5-7B-Instruct"
time = "00:30:00"
```

### `isambard_config.toml` — HPC cluster config

Configures the Isambard AI Phase 2 cluster connection. Located at `src/isambard_utils/assets/config.toml`.

```toml
[isambard]
project_dir = "/projects/a5u/ai-index-v2"
hf_cache_dir = "{project_dir}/hf_cache"
logs_dir = "{project_dir}/logs"
partition = "workq"
default_gpus = 1
default_cpus = 16
default_mem = "80G"
default_time = "12:00:00"
cuda_module = "cudatoolkit/24.11_12.6"
python_version = "3.12"
torch_index_url = "https://download.pytorch.org/whl/cu126"
```

### `netrun.json` — Pipeline graph

Defines the DAG, node_var placeholders, and cache settings. All node_vars (global and per-node) are declared with types only — no default values. Defaults live in `run_defs.toml`.

Key global node_vars: `sample_n`, `sample_seed`, `embedding_model`, `llm_model`, `cosine_mode`, `topk`, `llm_batch_size`, `llm_max_new_tokens`, `llm_max_concurrent_batches`, `run_name`, `adzuna_s3_prefix` (from `$env`).

### Adding new node variables

There are two kinds of node variables:

1. **Global node_vars** — Declared in `config/netrun.json` top-level `node_vars` with a type. Available to all nodes via `ctx.vars["var_name"]`. To add one:
   - Add `"var_name": {"type": "int"}` to `netrun.json` `node_vars`
   - Add `var_name = 1000` to `run_defs.toml` `[defaults]`

2. **Per-node vars** — Declared in a node's `execution_config.node_vars` in `netrun.json` with a type. To add one:
   - Add to the node's `execution_config.node_vars` in `netrun.json`:
     ```json
     "node_vars": { "my_var": { "type": "bool" } }
     ```
   - Add default to `run_defs.toml` in `[defaults.<node_name>]` subtable
   - Per-run overrides go in `[runs.<run_name>.<node_name>]` subtables
   - Access via `ctx.vars["my_var"]` in the node function

3. **Inherited per-node vars** — A per-node var can inherit its type, options, and default value from a global var of the same name using `"inherit": true`. The node can optionally override just the value. To add one:
   - Add to the node's `execution_config.node_vars` in `netrun.json`:
     ```json
     "node_vars": { "llm_model": { "inherit": true } }
     ```
   - The var inherits type/options from the global `llm_model`. No need to set type.
   - By default, the node gets the global value. To override, set `llm_model = "other-model"` in `[defaults.<node_name>]` in `run_defs.toml`.
   - Constraints: `inherit: true` must not set `type` or `options` (they come from the global). A global var with the same name must exist.

**Conventions:**
- `netrun.json` only declares names and types. All default values go in `run_defs.toml`.
- Node code must use `ctx.vars["var_name"]` (direct access), never `ctx.vars.get("var_name", default)`. Hidden defaults in code are a code smell — if a variable is missing, it should fail loudly so the missing config entry is noticed and fixed.
- Do not cast `ctx.vars` values (e.g. `int(ctx.vars["x"])`). The node_var type declarations in `netrun.json` handle type coercion. Only cast if netrun's type system doesn't support the desired type.

All node vars are accessible in node functions via `ctx.vars["var_name"]`. Values from TOML are Python-typed (int, str, bool) but may need explicit casting with `int()` when the type system returns strings.

## No Silent Fallbacks

**NEVER use `.get(key, default)` or fallback values when accessing data that is expected to exist.** Use direct key access (`d["key"]`) so that missing or malformed data fails immediately with a clear error. Silent fallbacks (empty strings, empty lists, `None`) hide bugs and produce subtly wrong results that are much harder to debug downstream.

This applies everywhere — not just `ctx.vars`, but also parsed JSON, dataframe columns, config dicts, and any structured data where the schema is known. If a field is required, access it directly and let `KeyError` surface the problem.

```python
# WRONG — silently produces garbage if schema changes
domain = parsed.get("domain", "")
tasks = parsed.get("tasks", [])

# RIGHT — fails loudly if the field is missing
domain = parsed["domain"]
tasks = parsed["tasks"]
```

## Execution Modes

Execution mode is determined per-model via the `mode` field in `embed_models.toml` / `llm_models.toml`. The pipeline's `embedding_model`, `llm_model`, and `cosine_mode` node_vars select which model config (and therefore which mode) to use.

| Mode | Description |
|------|-------------|
| `api` | No GPU needed: embeddings via OpenAI/Gemini API, cosine sim via numpy, LLM via `adulib.llm` (litellm) |
| `local` | Direct CUDA on current machine (sentence-transformers, torch). Mac variants use CPU/float32. |
| `sbatch` | Orchestrate from local: serialize inputs, submit SBATCH job to Isambard, wait, download results. Configurable `gpus` (default 1) for multi-GPU tensor parallelism. |

### Multi-GPU and FP8 support

Model TOML configs support `gpus` (number of GPUs) and `dtype = "fp8"` for sbatch mode:
- `gpus` controls both the SBATCH `--gpus=N` allocation and (for LLM) vLLM's `tensor_parallel_size`
- FP8 is the default dtype for sbatch LLM inference (native H100 tensor core support, ~2x memory savings)
- For embeddings/cosine, `gpus` only controls the SBATCH allocation (single-device models)
- `gpus` flows through: model TOML -> `_split_remote_kwargs` -> `arun_remote(gpus=N)` -> `SbatchConfig(gpus=N)`
- `tensor_parallel_size` flows through: `cfg["tensor_parallel_size"]` -> CLI `config_dict` -> `run_llm_generate` -> `load_llm` -> `_load_vllm`

Per-model GPU override example in `llm_models.toml`:
```toml
[models.llama-70b-sbatch]
mode = "sbatch"
model = "meta-llama/Llama-3-70B-Instruct"
gpus = 4
time = "04:00:00"
```

### Key files
- `pts/ai_index/utils/` — `embed()`, `llm_generate()`, `cosine_topk()`, `_load_model_config()`, `_resolve_model_args()`
- `config/embed_models.toml` — Embedding model configs
- `config/llm_models.toml` — LLM model configs

## `ai_index.utils` — Pipeline utilities

Model-key-based utility functions used by pipeline nodes. Each function resolves its execution mode from the TOML config, so callers only pass a model key.

### Model config resolution
- `_load_model_config(config_path, model_key)` — looks up `models.<key>`, reads `mode`, merges `defaults.<mode>` with model entry, returns `(mode, merged_dict)`
- `_resolve_model_args(config_path, model_key, kwargs)` — calls `_load_model_config`, merges `kwargs`, pops `model` name, returns `(mode, model_name, cfg)`

### Public functions
- **`embed(texts, *, model, **kwargs) -> np.ndarray`** — Embed texts. Routes by mode:
  - `api`: `adulib.llm.batch_embeddings` (litellm -> OpenAI/Gemini/etc.)
  - `local`: `llm_runner.embed.run_embeddings` (sentence-transformers, CUDA)
  - `sbatch`: `isambard_utils.orchestrate.run_remote("embed", ...)`
- **`aembed(texts, *, model, **kwargs) -> np.ndarray`** — Async version of `embed`
- **`llm_generate(prompts, *, model, **kwargs) -> list[str]`** — Generate LLM responses. Routes by mode:
  - `api`: `llm_runner.llm.run_llm_generate` with `backend="api"` (adulib -> litellm)
  - `local`: `llm_runner.llm.run_llm_generate` with `backend="transformers"` (CUDA)
  - `sbatch`: `isambard_utils.orchestrate.run_remote("llm_generate", ...)`
- **`allm_generate(prompts, *, model, **kwargs) -> list[str]`** — Async version
- **`cosine_topk(A, B, k, *, mode, **kwargs) -> dict`** — Top-K cosine similarity. Returns `{"indices": (n,k), "scores": (n,k)}`. Routes by mode:
  - `api`/`local`: `llm_runner.cosine.run_cosine_topk` (api uses `device="cpu"`)
  - `sbatch`: `isambard_utils.orchestrate.run_remote("cosine_topk", ...)`
- **`acosine_topk(A, B, k, *, mode, **kwargs) -> dict`** — Async version

Extra kwargs from the TOML config (e.g., `retry_delay`, `max_retries`) flow through to the underlying adulib/litellm calls. Explicit `**kwargs` passed to `llm_generate`/`embed` override TOML config values (via `cfg.update(kwargs)` in `_resolve_model_args`).

### `OnetScoreSet` (scoring.py)
Standard output format for O\*NET occupation-level score nodes. All score nodes (`score_presence`, `score_felten`, `score_task_exposure`) produce an `OnetScoreSet` — a validated DataFrame with `onet_code` + float score columns in [0, 1]. Provides `.validate()` (checks column types and ranges) and `.save(output_dir)` (writes `scores.csv`).

### `llm_generate` kwargs passthrough
All three backends (transformers `LLM`, `VllmLLM`, `ApiLLM`) support `system_message`, `max_new_tokens`, and `json_schema` in their `generate()` method. To use a system prompt from a node, pass it as a kwarg:
```python
llm_generate(prompts, model="gpt-5.2", system_message="You are...", max_new_tokens=220, json_schema=schema)
```
The `system_message` flows through: `llm_generate` → `run_llm_generate` → backend `.generate(system_message=...)`. Each backend handles it correctly (chat template for transformers, chat messages for vLLM, `system=` param for API/adulib).

### Structured JSON output (`json_schema`)
All three backends support constraining LLM output to valid JSON matching a schema. Pass a `json_schema` dict (from Pydantic's `model_json_schema()`) to guarantee parseable output:
- **transformers**: Uses [outlines](https://github.com/dottxt-ai/outlines) (`pip install outlines`). API: `outlines.from_transformers(model, tokenizer)` → `outlines.Generator(outlines_model, outlines.json_schema(schema))` → `generator.batch(texts, max_new_tokens=N)`
- **vLLM** (0.17.0): Uses `SamplingParams(structured_outputs=StructuredOutputsParams(json=json_schema))` from `vllm.sampling_params`
- **API**: Passes `response_format={"type": "json_schema", "json_schema": {"name": "response", "schema": json_schema}}` to adulib's `async_single`

See `pts/examples/02_structured_output.pct.py` for usage examples.

## Isambard HPC

The `isambard_utils` package automates interaction with the Isambard HPC cluster for GPU-intensive pipeline nodes.

**Cluster:** Isambard AI Phase 2 — NVIDIA GH200 120GB (ARM64), Slurm scheduler
**Project dir:** `/projects/a5u/ai-index-v2` (configured in `isambard_config.toml`)
**SSH host:** `ISAMBARD_HOST` env var in `.env` (Clifton certificate auth, 12hr renewal)

### isambard_utils modules
- `config` — `IsambardConfig` pydantic model, loads from `config.toml` + `.env`
- `ssh` — SSH command execution via subprocess (`run`/`arun` sync/async pairs)
- `transfer` — rsync upload/download, tar+SSH pipes. Three transfer modes: DIRECT (ephemeral stream), UPLOAD (content-hashed rsync), COMPRESSED (tar.gz stream)
- `slurm` — Slurm job submit/status/wait/cancel/log. Returns `SlurmJob` dataclass
- `env` — Remote environment bootstrap (ensure uv, create venv, install CUDA torch)
- `sbatch` — SBATCH script generation from `SbatchConfig` dataclass
- `models` — HuggingFace model pre-caching (`ensure_model`/`aensure_model`) + compute-node model loading (`load_embedding_model`, `load_llm`)
- `orchestrate` — High-level remote execution: `run_remote`/`arun_remote` + `setup_runner`/`asetup_runner`

### Remote execution flow (`arun_remote`)

The `orchestrate` module handles the full sbatch lifecycle:
1. **Setup** — deploy `llm_runner` to Isambard (idempotent)
2. **Pre-cache models** — download HuggingFace models to login node cache
3. **Transfer inputs** — serialize + upload via chosen transfer mode
4. **Submit SBATCH** — generate script, submit via `slurm.asubmit`
5. **Poll** — wait for Slurm job completion
6. **Download outputs** — retrieve + deserialize results
7. **Cleanup** — remove work directory (cache preserved)

### Running integration tests
```bash
pytest src/tests/isambard_utils/
```
Tests SSH, file transfer, env setup, GPU access, LLM inference, and job cancellation. Requires active Clifton cert.

## Project Structure

```
├── nblite.toml              # nblite config (export pipelines)
├── pyproject.toml            # Package config (ai-index)
├── .env                      # Environment variables (ISAMBARD_HOST, ADZUNA_S3_PREFIX, etc.)
├── config/                   # All configuration files
│   ├── netrun.json           # Pipeline graph with node_var placeholders
│   ├── run_defs.toml         # Run definitions (defaults + named runs)
│   ├── embed_models.toml     # Embedding model configs
│   └── llm_models.toml       # LLM model configs
├── prompt_library/           # Prompt templates (Markdown files: llm_summarise, llm_filter, score_task_exposure)
├── agent-context/            # Reference docs for netrun & nblite
├── pts/ai_index/             # Source of truth (.pct.py files) - EDIT THESE
│   ├── const.pct.py          # Path constants
│   ├── utils/                # embed(), llm_generate(), cosine_topk(), etc.
│   ├── run_pipeline.pct.py   # Pipeline runner
│   └── nodes/                # Node functions (17 nodes)
├── nbs/ai_index/             # Jupyter notebooks (auto-generated from pts)
├── src/ai_index/             # Python modules (auto-generated) - DO NOT EDIT
├── pts/isambard_utils/       # Isambard HPC utils (.pct.py) - EDIT THESE
├── nbs/isambard_utils/       # Isambard notebooks (auto-generated)
├── src/isambard_utils/       # Isambard utils Python modules (auto-generated)
│   └── assets/
│       └── config.toml       # Isambard cluster config
├── pts/llm_runner/           # LLM runner (embed, cosine, LLM generate) - EDIT THESE
├── nbs/llm_runner/           # LLM runner notebooks (auto-generated)
├── src/llm_runner/           # LLM runner Python modules (auto-generated)
├── pts/dev_utils/            # Development utilities (set_node_func_args, etc.)
├── nbs/dev_utils/            # Dev utils notebooks (auto-generated)
├── src/dev_utils/            # Dev utils Python modules (auto-generated)
├── pts/examples/             # Example notebooks
├── nbs/examples/             # Example notebooks (.ipynb, auto-generated)
├── pts/tests/                # Test notebooks (.pct.py) - EDIT THESE
│   ├── isambard_utils/       # Isambard integration + unit tests
│   └── llm_runner/           # LLM runner tests
├── nbs/tests/                # Test notebooks (.ipynb, auto-generated)
├── src/tests/                # Test modules (auto-generated)
└── .claude/skills/           # Netrun skill docs for Claude
```

## Development Workflow

### Where to edit code
- **Edit `.pct.py` files in `pts/`** - these are the source of truth
- **Never edit files in `src/`** - they are auto-generated and will be overwritten
- **Exception: `__init__.py` files** — nblite skips dunder-named files (`__init__`, `__main__`, etc.) during module export. These must be edited directly in `src/` and kept in sync with the corresponding `pts/.../__init__.pct.py` notebook.
- After editing `.pct.py` files, run: `nbl export --reverse && nbl export`

### nblite commands
```bash
nbl export                    # Export nbs -> pts -> src
nbl export --reverse          # Sync pts changes back to nbs
nbl test                      # Test notebooks execute without errors
nbl fill                      # Execute notebooks and save outputs
nbl new pts/ai_index/foo.pct.py  # Create new notebook
nbl new --template dev/templates/func_node.pct.py.jinja pts/ai_index/nodes/foo.pct.py  # Create new node notebook from template
```

### Export pipeline (from nblite.toml)
```
nbs -> lib        (nbs/ai_index/*.ipynb -> src/ai_index/*.py)
nbs -> pts        (nbs/ai_index/*.ipynb -> pts/ai_index/*.pct.py)
nbs_tests -> lib_tests          (nbs/tests/ -> src/tests/)
nbs_tests -> pts_tests          (nbs/tests/ -> pts/tests/)
nbs_isambard -> lib_isambard    (nbs/isambard_utils/ -> src/isambard_utils/)
nbs_isambard -> pts_isambard
nbs_dev -> lib_dev              (nbs/dev_utils/ -> src/dev_utils/)
nbs_dev -> pts_dev
nbs_examples -> pts_examples    (nbs/examples/ -> pts/examples/, no lib)
nbs_runner -> lib_runner        (nbs/llm_runner/ -> src/llm_runner/)
nbs_runner -> pts_runner
```

### Testing
```bash
pytest                        # Run tests from src/tests/
nbl test                      # Test all notebooks execute
```

### Key nblite directives for .pct.py files
- `#|default_exp module_name` - Set export target module (once per notebook, near top)
- `#|export` - Export cell to the Python module
- `#|exporti` - Export but exclude from `__all__`
- `#|top_export` - Export at module level, **outside** the generated function (for `export_as_func` notebooks). Use this for classes, constants, or imports that need to be importable from the module by other modules. Example: `JobInfoModel` in `llm_summarise` is `#|top_export` so `embed_ads` can `from ai_index.nodes.llm_summarise import JobInfoModel`.
- `#|hide` - Hide cell from docs
- `#|eval: false` - Skip cell during execution
- `#|export_as_func true` - Export entire notebook as a callable function
- `#|set_func_signature` - Define function name/params for function-export mode
- `#|func_return_line` - Inline on a line: converts that line to a `return` statement in the exported module. Avoids bare `return` (which would cause SyntaxError during `nbl fill`).
- `#|func_return` - Cell-level: prepends `return` to the first line of the cell.

### Important: `#|func_return_line` must be inside an `#|export` cell

The `#|func_return_line` directive **only works when the line is inside a cell marked with `#|export`**. If the return line is in a non-exported cell, it is silently dropped during module export — no error, no `return` statement in the generated `.py` file. This means the function will return `None`.

**Correct pattern** (return line in the same `#|export` cell):
```python
# %%
#|export
result = compute_something()
print(f"done: {result}")
result #|func_return_line
```

**Wrong pattern** (return line in a separate non-export cell — will be silently dropped):
```python
# %%
#|export
result = compute_something()

# %%
result #|func_return_line   # BUG: this won't appear in the generated module!
```

## Important: Suspected netrun bugs

If you encounter behaviour that looks like a bug in **netrun itself** (not in our node code), **stop immediately and notify the user**. Do not try to work around it. We may need to fix netrun upstream before continuing.

## Netrun Conventions

Data pipelines are defined as netrun graphs (JSON or TOML config) with Python node functions. See `agent-context/NETRUN_INSTRUCTIONS_CONCISE.md` for the full reference or use the `/netrun-*` skills for specific topics.

Key concepts:
- **Nodes**: Python functions that process data (defined via function factory)
- **Edges**: Connect output ports to input ports between nodes
- **Packets**: Units of data flowing through edges
- **Epochs**: One execution cycle of a node
- **Net**: The runtime that manages the graph

### Signals and Controls

Signals are output ports that fire automatically on lifecycle events. Controls are input ports that trigger node actions. Both are declared in `execution_config` and generate ports with `__signal_<type>__` / `__control_<type>__` naming.

**Valid signal types:** `epoch_started`, `epoch_finished`, `epoch_failed`, `epoch_cancelled`, `node_started`, `node_stopped`

**Valid control types:** `start_node`, `start_epoch`, `enable`, `disable`, `cancel_epoch`, `cancel_all_epochs`, `reset_epoch_count`, `set_epoch_count`, `stop_node`

Edge format: `"source_str": "nodeA.__signal_epoch_finished__"` → `"target_str": "nodeB.__control_start_epoch__"`

**Fan-out restriction:** Netrun forbids connecting one output port to multiple input ports. Use a **broadcast node** (`netrun.node_factories.broadcast`) for fan-out:

```json
{
  "name": "broadcast_my_signal",
  "factory": "netrun.node_factories.broadcast",
  "factory_args": { "num_outputs": 2 },
  "type": "node"
}
```

Ports: `in_0` (input), `out_0`, `out_1`, ... `out_{N-1}` (outputs). Each incoming packet is replicated to all output ports. Optional `copy_mode`: `"none"` (default, same reference), `"shallow"`, or `"deep"`.

**Fan-in / synchronization barrier:** Use a **join node** (`netrun.node_factories.join`) to wait for packets on multiple input ports before proceeding. The join fires once all ports have received their required packets, then emits a single dict on the `out` port.

```json
{
  "name": "join_scores",
  "factory": "netrun.node_factories.join",
  "factory_args": { "ports": ["presence", "felten", "task_exposure"] },
  "type": "node"
}
```

- **List form:** `"ports": ["a", "b", "c"]` — each port collects 1 packet. Output: `{"a": val, "b": val, "c": val}`.
- **Dict form:** `"ports": {"data": 1, "batch": 3}` — port "batch" collects 3 packets into a list. Output: `{"data": scalar, "batch": [v1, v2, v3]}`.
- Single output port: `out`.

### Validating Config Changes

**Always run `netrun validate` after modifying `netrun.json`:**

```bash
uv run netrun validate -c config/netrun.json
```

This catches fan-out violations, missing ports, invalid edges, and other graph errors before runtime. A valid config returns `{"valid": true, ...}`. An invalid config exits with code 1 and lists errors.

### Worker Pools

Pools control how node functions are executed. Configured at the net level in `pools`:

| Type | Config | Use case |
|------|--------|----------|
| `main` | `{"type": "main"}` | Single async worker (default if no pools defined) |
| `thread` | `{"type": "thread", "num_workers": N}` | Thread pool — supports both sync and async functions |
| `multiprocess` | `{"type": "multiprocess", "num_processes": N}` | CPU-bound work |
| `remote` | `{"type": "remote", "url": "ws://..."}` | Distributed execution |

**Thread pools support async functions.** When an async function runs in a thread pool, netrun uses `run_until_complete()` to execute it. This means all nodes (sync and async) can run in the same thread pool without issue.

**This project** overrides the default `"main"` pool with a thread pool (`"num_workers": 4`) so that sync I/O nodes (fetch_onet, fetch_adzuna, sample_ads, prepare_onet_targets) don't block the event loop. Async nodes (llm_summarise, embed_ads, embed_onet) also run in this pool via `run_until_complete()`.

There is no net-level setting to change the default pool assignment — all nodes default to `pools: ["main"]`. To use a different pool, set `"pools": ["pool_name"]` per-node in `execution_config`.

Pool config uses nested `spec` in JSON:
```json
"pools": {
  "main": {
    "spec": { "type": "thread", "num_workers": 4 }
  }
}
```

### Input Port Types and Batch Collection

In the `from_function` factory, parameter type annotations are purely type declarations. `list[int]` means "this port expects a single packet whose value is `list[int]`."

To collect **multiple packets** into a list (batch semantics), use `Batch` from `netrun.node_factories.from_function`:

```python
from netrun.node_factories.from_function import Batch

def process(data: Batch(str)):     # collects ALL packets into list[str]
    ...

def process(data: Batch(str, count=3)):  # collects up to 3 packets into list[str]
    ...
```

| Annotation | Salvo count | Function receives |
|---|---|---|
| `x: int` | 1 packet | `int` |
| `x: list[int]` | 1 packet | `list[int]` (single value) |
| `x: Batch(int)` | All packets | `list[int]` (collected) |
| `x: Batch(int, count=3)` | Up to 3 packets | `list[int]` (collected) |

**Do NOT use `list[T]` for batch collection** — it will not collect packets. Use `Batch(T)` instead.

### Key Net APIs
- **`run_to_targets(targets)`** — Run upstream nodes and collect input salvos at target. Auto-starts the Net if not started. Executes all source nodes (`run_on_startup=True`) automatically. Returns `list[TargetInputSalvo]` with `.packets: dict[str, list[Any]]`.
- **`inject_data(node_name, port, values)`** — Inject data into a node's input port. Works before `Net.start()`.
- **`on_epoch_start(callback)` / `on_epoch_end(callback)`** — Register lifecycle callbacks. Callback signature: `(node_name, epoch_id)` for start, `(node_name, epoch_id, record)` for end. Returns a `remove()` callable. Also available on `NodeInfo` (fires only for that node).
- **`Net(config, run_source_nodes=True)`** — Constructor. `run_source_nodes` (formerly `run_startup_nodes`) controls whether source nodes execute during `start()`.

## Old Repository Reference

The old codebase is at `/Users/lukas/dev/20260208_e22t36__aisi-economy-index`. Key locations:
- `nbs/helpers/` - Core data processing notebooks
- `nbs/isambard/2026_01/` - GPU pipeline (3-stage: embed, cosine, LLM filter)
- `nbs/api/` - Configuration and CLI modules
- `aisi_economy_index/assets/config.toml` - Path configuration
- `nbs/__scratch/` - Experimental/legacy (ignore unless needed)

Old dependencies of note: `adulib[llm]` (LLM abstraction), `sentence-transformers`, `google-generativeai`, `polars`, `pandas`.
