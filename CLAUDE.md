# CLAUDE.md

## Project Overview

**ai-index** (AISI Economy Index v2) is a productionized data pipeline for analyzing AI exposure in the economy. It matches job advertisements to O\*NET occupations and computes AI impact metrics (ASPECTT vectors, AI exposure scores, seniority/job zone).

This is a clean rewrite of the old repository at `/Users/lukas/dev/20260208_e22t36__aisi-economy-index`, which was a collection of manually-run notebooks. The v2 uses **netrun** for orchestrating the data pipeline and **nblite** for literate programming development.

## Pipeline DAG

The pipeline is defined across `src/ai_index/assets/netrun.json` (parent) and 7 subgraph files in `src/ai_index/assets/subgraphs/`. Subgraphs are flattened at resolution time — node names get prefixed (e.g., `embed_onet` → `job_ad_matching.embed_onet`).

```
  ┌─────────────────┐
  │ [data_prep]      │
  │  fetch_onet      │
  │  load_job_ads    │
  └──┬───────────┬───┘
     │onet_tables│job_ads
     │           │
     │       bc_job_ads (parent)
     │        ├──┬───────────┐
     ▼        │  │           │
  ┌──────────────────────────────────────────────────────┐
  │ [exposure_scores]                                     │
  │  bc_onet_tables ─┬► build_onet_descriptions           │
  │                  └► build_onet_eval_dfs               │
  │                       │         │        │            │
  │                       ▼         ▼        ▼            │
  │              score_task_exp   score_    score_         │
  │                    │         presence   felten         │
  │                    └────┬───────┘──────┘              │
  │                         ▼                             │
  │                aggregate_soc_exposure                  │
  └──┬──────────────────────┬─────────────────────────────┘
     │descriptions          │exposure_scores
     ▼                      │
  ┌──────────────────────┐  │  bc_exposure_scores (parent)
  │ [job_ad_matching]    │  │   ├──────────────┐
  │  bc_desc ─► embed_   │  │   │              │
  │  job_ads ─► onet     │  │   ▼              ▼
  │             │        │  │  ┌──────────┐  ┌─────────────────┐
  │             ▼        │  │  │[benchmark│  │ [generate_index] │
  │          cos_sim     │  │  │_exposure]│  │  combine_job_    │
  │             │        │  │  │benchmark │  │  exposure ◄──────┤
  │             ▼        │  │  │_exposure │  └────────┬─────────┘
  │          llm_filter  │  │  └──────────┘           │
  └──────────┬───────────┘  │              job_exposure_index
             │weighted_codes│                         │
             └──────────────┼─────────► [generate_    │
                            └────────►    index]      │
                                                      ▼
                                         ┌────────────────────────┐
                                         │ [index_analysis]        │
                                         │  bc_job_exp ─┬► agg_geo│
                                         │              └► summary│
                                         └────────────────────────┘
```

### Parent graph (2 nodes + 7 subgraphs, 11 edges)
- `bc_job_ads` — Fan out job_ads to matching (×2) and generate_index (×1)
- `bc_exposure_scores` — Fan out to benchmark (×1) and generate_index (×1)
- 7 subgraphs: `data_prep`, `exposure_scores`, `job_ad_matching`, `benchmark_exposure_scores`, `benchmark_job_ad_matching`, `generate_index`, `index_analysis`

### `data_prep` subgraph (2 nodes, 0 edges)
- `fetch_onet` (run_on_startup) — Download O\*NET 30.0 database
- `load_job_ads` (run_on_startup) — Load job advertisement dataset
- Exposed out: `onet_tables`, `job_ads`

### `exposure_scores` subgraph (8 nodes, 10 edges)
- `bc_onet_tables` → `build_onet_descriptions` + `build_onet_eval_dfs`
- `score_task_exposure` — GPT task-level 3-level scoring, aggregated to SOC
- `score_presence` — Humanness scoring (physical, emotional, creative)
- `score_felten` — Felten ability exposure scoring (multiple scenarios)
- `aggregate_soc_exposure` — Merge + normalize all score types at SOC level
- Exposed in: `onet_tables`; out: `descriptions`, `exposure_scores`

### `job_ad_matching` subgraph (5 nodes, 5 edges)
- `embed_onet` — Embed O\*NET occupations with BGE-large (894×1024 float16)
- `embed_job_ads` — Embed job ads with BGE-large (N×1024 float16)
- `compute_cosine_similarity` — Top-K candidate matches (top-5 role + top-5 task)
- `llm_filter_candidates` — LLM negative selection → normalized weights per job
- Exposed in: `descriptions`, `job_ads_embed`, `job_ads_llm`; out: `weighted_codes`

### `benchmark_exposure_scores` subgraph (1 node, 0 edges)
- `benchmark_exposure` — Benchmarking stats for exposure scores
- Exposed in: `exposure_scores`; out: `benchmark`

### `benchmark_job_ad_matching` subgraph (placeholder, 0 nodes)

### `generate_index` subgraph (1 node, 0 edges)
- `combine_job_exposure` — Weighted sum of matched exposure scores → per-job AI exposure
- Exposed in: `weighted_codes`, `exposure_scores`, `job_ads`; out: `job_exposure_index`

### `index_analysis` subgraph (3 nodes, 2 edges)
- `aggregate_geography` — Aggregate by geographic dimensions
- `compute_summary_stats` — Summary statistics and visualizations
- Exposed in: `job_exposure_index`; out: `geography_index`, `summary`

### Node Function Paths

Each node is a module at `ai_index.nodes.<name>` (developed as `pts/ai_index/nodes/<name>.pct.py`).

### Old Pipeline Reference

The old pipeline (now superseded by the DAG above) had four stages:
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
- **torch** - GPU inference (LLaMA 3.1-8B)
- **pandas / polars** - Data manipulation
- **pydantic** - Configuration and data validation
- **isambard_utils** - Isambard HPC interaction (SSH, rsync, Slurm, env setup)
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
- `src/ai_index/assets/run_defs.toml` — Run definitions
- `src/ai_index/assets/netrun.json` — Pipeline graph with unfilled node_var placeholders

## Configuration Files

### `run_defs.toml` — Run definitions

Defines named pipeline configurations. `[defaults]` provides base values; `[runs.<name>]` overrides them. All values are injected as netrun global node_vars (accessible via `ctx.vars` in nodes).

```toml
[defaults]
years = "2025"           # Comma-delimited year filter ("" = all years)
sample_n = 0             # 0 = full run, N = sample N ads
sample_seed = 42
embedding_model = "text-embedding-3-large"   # Key into embed_models.toml
cosine_mode = "api"      # "api", "local", or "sbatch"
topk = 5                 # Top-K candidates for cosine similarity
llm_model = "gpt-5.2"   # Key into llm_models.toml

[runs.baseline]          # Inherits all defaults
[runs.test]              # Quick test (10 ads, otherwise defaults)
sample_n = 10
[runs.test_api]          # API mode test
[runs.test_local]        # Local GPU mode test
[runs.test_sbatch]       # Isambard sbatch mode test
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
job_name = "embed"
time = "01:00:00"
setup = false

[models."text-embedding-3-large"]
mode = "api"
model = "text-embedding-3-large"
[models.bge-large-local]
mode = "local"
model = "BAAI/bge-large-en-v1.5"
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
dtype = "float16"
backend = "vllm"
job_name = "llm_generate"
time = "02:00:00"
setup = false

[models."gpt-5.2"]
mode = "api"
model = "openai/gpt-5.2"
[models.qwen-7b-local]
mode = "local"
model = "Qwen/Qwen2.5-7B-Instruct"
[models.qwen-7b-sbatch]
mode = "sbatch"
model = "Qwen/Qwen2.5-7B-Instruct"
time = "00:30:00"
```

### `isambard_config.toml` — HPC cluster config

Configures the Isambard AI Phase 2 cluster connection. Symlinked from `src/isambard_utils/assets/config.toml`.

```toml
[isambard]
project_dir = "/projects/a5u/ai-index-v2"
hf_cache_dir = "{project_dir}/hf_cache"
partition = "workq"
default_gpus = 1
default_cpus = 16
default_mem = "80G"
default_time = "12:00:00"
cuda_module = "cudatoolkit/24.11_12.6"
python_version = "3.12"
```

### `netrun.json` — Pipeline graph

Defines the DAG, node_var placeholders, and cache settings. Global node_vars are declared with types but no values (filled at runtime by run_defs). Per-node vars use `"inherit": true` to pull from globals.

Key global node_vars: `years`, `sample_n`, `sample_seed`, `embedding_model`, `llm_model`, `cosine_mode`, `topk`, `run_name`, `adzuna_s3_prefix` (from `$env`).

## Execution Modes

Execution mode is determined per-model via the `mode` field in `embed_models.toml` / `llm_models.toml`. The pipeline's `embedding_model`, `llm_model`, and `cosine_mode` node_vars select which model config (and therefore which mode) to use.

| Mode | Description |
|------|-------------|
| `api` | No GPU needed: embeddings via OpenAI API, cosine sim via numpy, LLM via `adulib.llm.async_single` |
| `local` | Direct CUDA on current machine (sentence-transformers, torch) |
| `sbatch` | Orchestrate from local: serialize inputs, submit SBATCH job to Isambard, wait, download results |

### Node structure pattern

GPU nodes (`embed_onet`, `embed_job_ads`, `compute_cosine_similarities`, `llm_filter_candidates`) follow this pattern:
1. Read model key from `ctx.vars` (e.g., `embedding_model`)
2. Call utility function with model key — mode is resolved from the TOML config
3. For sbatch mode: `maybe_run_remote()` guard handles the full remote lifecycle

### Key files
- `pts/ai_index/utils.pct.py` — `embed()`, `llm_generate()`, `cosine_topk()`, `_load_model_config()`, `_resolve_model_args()`
- `src/ai_index/assets/embed_models.toml` — Embedding model configs
- `src/ai_index/assets/llm_models.toml` — LLM model configs

## `ai_index.utils` — Pipeline utilities

Model-key-based utility functions used by pipeline nodes. Each function resolves its execution mode from the TOML config, so callers only pass a model key.

### Model config resolution
- `_load_model_config(config_path, model_key)` — looks up `models.<key>`, reads `mode`, merges `defaults.<mode>` with model entry, returns `(mode, merged_dict)`
- `_resolve_model_args(config_path, model_key, kwargs)` — calls `_load_model_config`, merges `kwargs`, pops `model` name, returns `(mode, model_name, cfg)`

### Public functions
- **`embed(texts, *, model, **kwargs) -> np.ndarray`** — Embed texts. Routes by mode:
  - `api`: `adulib.llm.batch_embeddings` (litellm → OpenAI/Gemini/etc.)
  - `local`: `llm_runner.embed.run_embeddings` (sentence-transformers, CUDA)
  - `sbatch`: `isambard_utils.orchestrate.run_remote("embed", ...)`
- **`aembed(texts, *, model, **kwargs) -> np.ndarray`** — Async version of `embed`
- **`llm_generate(prompts, *, model, **kwargs) -> list[str]`** — Generate LLM responses. Routes by mode:
  - `api`: `llm_runner.llm.run_llm_generate` with `backend="api"` (adulib → litellm)
  - `local`: `llm_runner.llm.run_llm_generate` with `backend="transformers"` (CUDA)
  - `sbatch`: `isambard_utils.orchestrate.run_remote("llm_generate", ...)`
- **`allm_generate(prompts, *, model, **kwargs) -> list[str]`** — Async version
- **`cosine_topk(A, B, k, *, mode, **kwargs) -> dict`** — Top-K cosine similarity. Returns `{"indices": (n,k), "scores": (n,k)}`. Routes by mode:
  - `api`/`local`: `llm_runner.cosine.run_cosine_topk` (api uses `device="cpu"`)
  - `sbatch`: `isambard_utils.orchestrate.run_remote("cosine_topk", ...)`
- **`acosine_topk(A, B, k, *, mode, **kwargs) -> dict`** — Async version

Extra kwargs from the TOML config (e.g., `retry_delay`, `max_retries`) flow through to the underlying adulib/litellm calls.

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
python -m isambard_utils_tests.test_integration
```
Tests SSH, file transfer, env setup, GPU access, LLM inference, and job cancellation. Requires active Clifton cert.

## Project Structure

```
├── nblite.toml              # nblite config (export pipelines)
├── pyproject.toml            # Package config (ai-index)
├── .env                      # Environment variables (ISAMBARD_HOST, ISAMBARD_PROJECT_DIR)
├── isambard_config.toml      # Symlink to src/isambard_utils/assets/config.toml
├── agent-context/            # Reference docs for netrun & nblite
├── pts/ai_index/             # Source of truth (.pct.py files) - EDIT THESE
├── nbs/ai_index/             # Jupyter notebooks (auto-generated from pts)
├── src/ai_index/             # Python modules (auto-generated) - DO NOT EDIT
│   └── assets/
│       ├── netrun.json       # Netrun parent graph (subgraph references + cross-subgraph edges)
│       └── subgraphs/        # 7 subgraph definitions (data_prep, exposure_scores, job_ad_matching, ...)
├── pts/isambard_utils/       # Isambard HPC utils (.pct.py) - EDIT THESE
├── src/isambard_utils/       # Isambard utils Python modules (auto-generated)
│   └── assets/
│       └── config.toml       # Isambard cluster config (project_dir, partition, etc.)
├── pts/isambard_utils_tests/ # Isambard integration tests (.pct.py) - EDIT THESE
├── src/isambard_utils_tests/ # Isambard test modules (auto-generated)
├── pts/tests/                # Test notebooks (.pct.py)
├── nbs/tests/                # Test notebooks (.ipynb, auto-generated)
├── src/tests/                # Test modules (auto-generated)
└── .claude/skills/           # Netrun skill docs for Claude
```

## Development Workflow

### Where to edit code
- **Edit `.pct.py` files in `pts/ai_index/`** - these are the source of truth
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
```

### Export pipeline (from nblite.toml)
```
nbs -> lib        (nbs/ai_index/*.ipynb -> src/ai_index/*.py)
nbs -> pts        (nbs/ai_index/*.ipynb -> pts/ai_index/*.pct.py)
nbs_tests -> lib_tests
nbs_tests -> pts_tests
nbs_isambard -> lib_isambard        (isambard_utils package)
nbs_isambard -> pts_isambard
nbs_isambard_tests -> lib_isambard_tests
nbs_isambard_tests -> pts_isambard_tests
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
- `#|top_export` - Export at module level (for function-export notebooks)
- `#|hide` - Hide cell from docs
- `#|eval: false` - Skip cell during execution
- `#|export_as_func true` - Export entire notebook as a callable function
- `#|set_func_signature` - Define function name/params for function-export mode

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

## Old Repository Reference

The old codebase is at `/Users/lukas/dev/20260208_e22t36__aisi-economy-index`. Key locations:
- `nbs/helpers/` - Core data processing notebooks
- `nbs/isambard/2026_01/` - GPU pipeline (3-stage: embed, cosine, LLM filter)
- `nbs/api/` - Configuration and CLI modules
- `aisi_economy_index/assets/config.toml` - Path configuration
- `nbs/__scratch/` - Experimental/legacy (ignore unless needed)

Old dependencies of note: `adulib[llm]` (LLM abstraction), `sentence-transformers`, `google-generativeai`, `polars`, `pandas`.
