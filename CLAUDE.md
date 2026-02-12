# CLAUDE.md

## Project Overview

**ai-index** (AISI Economy Index v2) is a productionized data pipeline for analyzing AI exposure in the economy. It matches job advertisements to O\*NET occupations and computes AI impact metrics (ASPECTT vectors, AI exposure scores, seniority/job zone).

This is a clean rewrite of the old repository at `/Users/lukas/dev/20260208_e22t36__aisi-economy-index`, which was a collection of manually-run notebooks. The v2 uses **netrun** for orchestrating the data pipeline and **nblite** for literate programming development.

## Pipeline DAG

The pipeline is defined in `src/ai_index/assets/netrun.json` (16 nodes, 21 edges). Two independent branches converge at the combine step:

```
          fetch_onet ─────────────────────────────── build_onet_eval_dfs
              │                                       │         │        │
              ▼                                       ▼         ▼        ▼
    build_onet_descriptions              score_task_exposure  score_  score_
              │         │                       │           presence  felten
              │         ▼                       │              │        │
              │     embed_onet                  ▼              ▼        ▼
              │         │                 aggregate_soc_exposure
              │         │                       │          │
              ▼         ▼                       │          ▼
load_job_ads ─┬─► embed_job_ads                 │   benchmark_exposure
              │         │                       │
              │         ▼                       │
              │  compute_cosine_similarity      │
              │         │                       │
              │         ▼                       ▼
              ├──► llm_filter_candidates        │
              │         │                       │
              │         ▼                       ▼
              └──► combine_job_exposure ◄───────┘
                        │
                   ┌────┴────┐
                   ▼         ▼
          aggregate_    compute_
          geography     summary_stats
```

### Node Inventory

**Data Preparation (4 nodes):**
- `fetch_onet` — Download O\*NET 30.0 database
- `build_onet_descriptions` — Build occupation descriptions (894 rows: SOC, title, description, tasks/skills text)
- `build_onet_eval_dfs` — Build eval DataFrames for exposure scoring (skills, abilities, knowledge, tasks, work context)
- `load_job_ads` — Load job advertisement dataset

**Matching Pipeline (4 nodes) → Weighted O\*NET codes per job:**
- `embed_onet` — Embed O\*NET occupations with BGE-large (894x1024 float16)
- `embed_job_ads` — Embed job ads with BGE-large (Nx1024 float16)
- `compute_cosine_similarity` — Top-K candidate matches (top-5 role + top-5 task, averaged overlaps)
- `llm_filter_candidates` — LLM negative selection (LLaMA-3.1-8B) → normalized weights per job

**Exposure Scoring (4 nodes) → AI exposure per O\*NET code:**
- `score_task_exposure` — GPT task-level 3-level scoring, aggregated to SOC
- `score_presence` — Humanness scoring across 3 dimensions (physical, emotional, creative)
- `score_felten` — Felten ability exposure scoring (multiple scenarios)
- `aggregate_soc_exposure` — Merge + normalize all score types at SOC level

**Benchmarking (1 node, side branch):**
- `benchmark_exposure` — Benchmarking stats for exposure scores

**Combination + Output (3 nodes):**
- `combine_job_exposure` — Weighted sum of matched O\*NET exposure scores → per-job AI exposure
- `aggregate_geography` — Aggregate by geographic dimensions
- `compute_summary_stats` — Summary statistics and visualizations

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

## Execution Modes

The 4 matching pipeline GPU nodes (`embed_onet`, `embed_job_ads`, `compute_cosine_similarity`, `llm_filter_candidates`) support configurable execution via the `execution_mode` node_var (env: `EXECUTION_MODE`, default: `local`):

| Mode | Description |
|------|-------------|
| `local` | Direct CUDA on current machine |
| `deploy` | Functionally identical to local (pipeline runs on Isambard directly) |
| `sbatch` | Orchestrate from local: serialize inputs, submit SBATCH job to Isambard, wait, download results |
| `api` | No GPU needed: embeddings via sentence-transformers on CPU, cosine sim via numpy, LLM via `adulib.llm.async_single` |

### Node structure pattern

Each GPU node follows this pattern:
1. **sbatch guard** (`maybe_run_remote()`) — if sbatch mode, handles full remote lifecycle and returns
2. **mode-aware body** — reads `execution_mode` from `ctx.vars`, sets `device` accordingly
3. For `compute_cosine_similarity`: api mode uses numpy-only CPU path (no torch import)
4. For `llm_filter_candidates`: api mode uses `adulib.llm.async_single()` + `batch_executor`

### Key files
- `pts/ai_index/utils.pct.py` — `ExecutionMode` type, `serialize_node_data`/`deserialize_node_data`, `maybe_run_remote()` guard, `_run_sbatch()` orchestration
- `test_matching_pipeline.py` — Synthetic CPU test of all 4 matching nodes

## Isambard HPC

The `isambard_utils` package automates interaction with the Isambard HPC cluster for GPU-intensive pipeline nodes.

**Cluster:** Isambard AI Phase 2 — NVIDIA GH200 120GB (ARM64), Slurm scheduler
**Project dir:** `/projects/a5u/ai-index-v2` (configured in `isambard_config.toml`)
**SSH host:** `ISAMBARD_HOST` env var in `.env` (Clifton certificate auth, 12hr renewal)

### isambard_utils modules
- `config` — `IsambardConfig` pydantic model, loads from `config.toml` + `.env`
- `ssh` — SSH command execution via subprocess
- `transfer` — rsync upload/download, SSH pipe for small data
- `slurm` — Slurm job submit/status/wait/cancel/log
- `env` — Remote environment bootstrap (uv, venv, code sync)
- `sbatch` — SBATCH script generation from `SbatchConfig`

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
│       └── netrun.json       # Netrun graph definition for the data pipeline
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
