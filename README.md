# AISI Exposure Index

A data pipeline for measuring AI exposure across the UK economy. It matches ~30 million Adzuna job advertisements to 861 O\*NET occupations using embedding similarity, LLM filtering, and cross-encoder reranking, then computes multi-dimensional AI exposure scores and aggregates them by geography.

[View interactive pipeline graph](https://htmlpreview.github.io/?https://github.com/Autonomy-Data-Unit/aisi-exposure-index/blob/main/pipeline.html)

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.sample .env   # then fill in credentials
```

Required environment variables (see `.env.sample`):

| Variable | Purpose |
|----------|---------|
| `ADZUNA_S3_PREFIX` | S3 path to Adzuna job ad data |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS credentials for S3 access |
| `HF_TOKEN` | Hugging Face token (for gated models) |
| `ISAMBARD_HOST` | Isambard HPC hostname (for sbatch mode) |
| `ZEROENTROPY_API_KEY` | ZeroEntropy API key (for zerank reranker) |

## Running the pipeline

```bash
uv run run-pipeline [RUN_NAME]
```

The run name selects a configuration from `config/run_defs.toml`. If omitted, defaults to `baseline` (which uses the `[defaults]` section: 10 sample ads, sbatch models).

### Available runs

| Run | Sample size | Models | Purpose |
|-----|-------------|--------|---------|
| *(default)* | 10 ads | bge-large + qwen-7b (sbatch) | Quick test with HPC |
| `test_api` | 10 ads | text-embedding-3-large + gpt-5.2 (API) | Test with OpenAI API |
| `test_local` | 10 ads | bge-large + qwen-0.5b (CPU) | Test locally, no GPU |
| `test_sbatch` | 10 ads | bge-large + qwen-7b (sbatch) | Test on Isambard HPC |
| `validation_5k` | 5,000 ads | sbatch models | Validation experiments |
| `benchmark_5k` | 5,000 ads | gpt-5.4 + text-embedding-3-large + zerank-2 | Frontier model benchmark |
| `calibration` | 1,000 ads | configurable | GPU-hours estimation |

Run definitions are composable: each named run inherits from `[defaults]` and overrides specific values. See `config/run_defs.toml` for the full list.

## Pipeline overview

The pipeline is a 17-node DAG orchestrated by [netrun](https://github.com/lukastk/netrun). It has three stages that run partly in parallel. See the [interactive pipeline graph](https://htmlpreview.github.io/?https://github.com/Autonomy-Data-Unit/aisi-exposure-index/blob/main/pipeline.html) for the full DAG visualization.

### Stage 1: Data ingestion

- **`fetch_adzuna`** / **`fetch_onet`**: Download source data (Adzuna ads from S3, O\*NET database). Run in parallel.
- **`sample_ads`**: Sample N ads for processing (or pass all through with `sample_n = -1`).
- **`prepare_onet_targets`**: Build rich text descriptions for each O\*NET occupation from titles, tasks, skills, and work activities.

### Stage 2a: Job ad matching

Matches each job ad to O\*NET occupations through a multi-stage retrieval pipeline:

1. **`embed_ads`** / **`embed_onet`**: Embed ad text and O\*NET descriptions using the configured embedding model.
2. **`cosine_candidates`**: Top-k cosine similarity between ad and O\*NET embeddings (default k=20).
3. **`llm_filter_candidates`**: LLM-based filtering. Selects candidates that are functional matches for the job ad.
4. **`rerank_candidates`**: Cross-encoder reranking of filtered candidates to produce final match scores.

### Stage 2b: O\*NET exposure scoring

Runs in parallel with ad matching. Computes three independent AI exposure dimensions per occupation:

- **`score_presence`**: Humanness/presence scores (physical, emotional, creative) from O\*NET work context data.
- **`score_felten`**: Felten AIOE ability-application AI exposure scores.
- **`score_task_exposure`**: LLM-based task-level AI exposure classification (3-level scale).
- **`combine_onet_exposure`**: Merges all scores into a single combined table.

### Stage 3: Index construction

- **`compute_job_ad_exposure`**: Maps occupation-level scores to individual ads via rerank-score-weighted averaging.
- **`aggregate_geo`**: Aggregates ad-level exposure scores by Local Authority District (LAD22CD).

## Execution modes

Each model (embedding, LLM, reranker) has a `mode` configured in `config/embed_models.toml`, `config/llm_models.toml`, or `config/rerank_models.toml`:

| Mode | Description |
|------|-------------|
| `api` | Remote API (OpenAI, Gemini, ZeroEntropy). No GPU needed. |
| `local` | Direct inference on the current machine (CUDA or CPU). |
| `sbatch` | Submit Slurm jobs to Isambard HPC, poll for completion, download results. |

The pipeline's `embedding_model`, `llm_model`, and `rerank_model` variables select which model key to use. The model key determines the execution mode.

## Configuration

All configuration lives in `config/`:

| File | Purpose |
|------|---------|
| `netrun.json` | Pipeline DAG (nodes, edges, node variable declarations) |
| `run_defs.toml` | Run definitions (default values and named run overrides) |
| `embed_models.toml` | Embedding model configs (mode, model name, parameters) |
| `llm_models.toml` | LLM model configs |
| `rerank_models.toml` | Reranker model configs |
| `deploy.toml` | Remote deployment settings (Hetzner Cloud server) |
| `prompt_library/` | LLM prompt templates (Markdown files) |

### How run definitions work

`config/run_defs.toml` has a `[defaults]` section and named `[runs.<name>]` sections. When you run `uv run run-pipeline my_run`, the defaults are merged with `[runs.my_run]` overrides. Scalar values become global node variables; subtable dicts (e.g. `[defaults.embed_ads]`) become per-node overrides. All values are accessible in node code via `ctx.vars["var_name"]`.

## Remote deployment

The pipeline can be deployed to a Hetzner Cloud server for remote execution:

```bash
uv run remote-deploy-pipeline          # Provision server + deploy code (idempotent)
uv run remote-run-pipeline <run_name>  # Run pipeline in background
uv run remote-bg-log --follow          # Tail the log
uv run remote-download-store <path> .  # Download results
uv run remote-destroy                  # Delete server
```

See `src/deploy/` for implementation. Server settings are in `config/deploy.toml`.

## Calibration

Estimate GPU-hours for full-scale runs on Isambard:

```bash
uv run run-calibration <llm_model_key> <embedding_model_key>
uv run estimate-calibration [N_ADS]   # default: 30,000,000
```

## Validation

Run multi-model validation experiments to measure agreement across model combinations:

```bash
uv run run-validation <run_def> <llm_model> <embed_model> [rerank_model]
uv run validate-all              # Run all configured combinations
uv run generate-reports          # Generate analysis reports
```

## Testing

```bash
uv run pytest          # Unit tests (src/tests/)
uv run nbl test        # Notebook execution tests
```

## Project structure

```
config/                          # All configuration
├── netrun.json                  #   Pipeline DAG definition
├── run_defs.toml                #   Run definitions and defaults
├── embed_models.toml            #   Embedding model configs
├── llm_models.toml              #   LLM model configs
├── rerank_models.toml           #   Reranker model configs
├── deploy.toml                  #   Remote deployment settings
└── prompt_library/              #   LLM prompt templates

src/                             # Python source (edit directly)
├── ai_index/                    #   Main pipeline package
│   ├── const.py                 #     Path constants
│   ├── run_pipeline.py          #     Pipeline runner entry point
│   ├── utils/                   #     embed(), llm_generate(), cosine_topk(), etc.
│   └── nodes/                   #     Node modules (auto-generated from pts/)
├── isambard_utils/              #   Isambard HPC interaction (SSH, Slurm, rsync)
├── llm_runner/                  #   Model inference backends
├── calibration/                 #   GPU-hours calibration tools
├── deploy/                      #   Remote deployment (Hetzner Cloud)
├── validation/                  #   Multi-model validation framework
├── dev_utils/                   #   Development utilities
└── tests/                       #   Test suite

pts/                             # Notebook source (.pct.py, nblite-managed)
├── ai_index/nodes/              #   Pipeline node notebooks (source of truth for nodes)
├── scratch/                     #   Experiments and examples
└── validation/                  #   Validation analysis notebooks

nbs/                             # Jupyter notebooks (auto-generated from pts/)
├── ai_index/nodes/
├── scratch/
└── validation/

store/                           # Data storage (gitignored)
├── inputs/                      #   Source data (O*NET, Adzuna DuckDB)
├── pipeline/{run_name}/         #   Run-specific intermediates
└── outputs/{run_name}/          #   Final outputs
```

## Development with nblite

[nblite](https://github.com/lukastk/nblite) provides literate programming for a subset of the codebase. Notebooks are authored as `.pct.py` files (percent-format scripts) in `pts/`, synced to `.ipynb` in `nbs/`, and exported as Python modules to `src/`.

Only three directories are managed by nblite (defined in `nblite.toml`):

| Location | Purpose | Exports to `src/`? |
|----------|---------|-------------------|
| `pts/ai_index/nodes/` | Pipeline node functions | Yes (`src/ai_index/nodes/`) |
| `pts/validation/` | Validation analysis notebooks | Yes (`src/validation/`) |
| `pts/scratch/` | Experiments and examples | No (pts/nbs sync only) |

Everything else in `src/` is plain Python, edited directly.

### Editing nodes

Pipeline nodes are developed as notebooks in `pts/ai_index/nodes/`. After editing:

```bash
uv run nbl export --reverse && uv run nbl export
```

This syncs: `pts/*.pct.py` <-> `nbs/*.ipynb` -> `src/*.py`.

### Key nblite directives

Directives are comments in `.pct.py` cells that control how code is exported:

- `#|default_exp module_name` -- set the target module name
- `#|export` -- export this cell to the Python module
- `#|export_as_func true` -- wrap the entire notebook into a single function
- `#|top_export` -- export at module level (outside the generated function)
- `#|func_return_line` -- inline on a line to make it a `return` statement in the exported module

## Pipeline orchestration with netrun

[netrun](https://github.com/lukastk/netrun) is a flow-based data pipeline framework. The pipeline DAG is defined in `config/netrun.json`.

Key concepts:

- **Nodes**: Python functions registered via a function factory. Each node has typed input/output ports.
- **Edges**: Connect an output port of one node to an input port of another. Data flows as packets.
- **Signals/Controls**: Lifecycle events (e.g. `epoch_finished`) trigger control actions (e.g. `start_epoch`) on downstream nodes.
- **Epochs**: One execution cycle of a node. The pipeline runs until no more progress can be made.
- **Node variables**: Typed configuration values declared in `netrun.json` and populated from `run_defs.toml`. Accessed via `ctx.vars["name"]` in node code.

### Editing the pipeline graph

Use [netrun-ui](https://github.com/lukastk/netrun) to visually edit `config/netrun.json`:

```bash
uv run netrun-ui config/netrun.json
```

### Useful netrun CLI commands

```bash
uv run netrun validate -c config/netrun.json   # Validate the DAG
uv run netrun nodes -c config/netrun.json      # List all nodes and ports
uv run netrun node -c config/netrun.json <name> # Detailed info about a node
uv run netrun structure -c config/netrun.json  # Output graph topology as JSON
```

## CLI reference

| Command | Description |
|---------|-------------|
| `uv run run-pipeline [RUN_NAME]` | Run the pipeline |
| `uv run clean-store` | Clean pipeline store data |
| `uv run run-calibration <llm> <embed>` | Calibrate GPU-hours |
| `uv run estimate-calibration [N]` | Estimate GPU-hours for N ads |
| `uv run calibrate-all` | Calibrate all uncalibrated model combinations |
| `uv run run-validation ...` | Run a validation experiment |
| `uv run validate-all` | Run all configured validation combinations |
| `uv run generate-reports` | Generate validation reports |
| `uv run publish-reports` | Publish validation reports |
| `uv run remote-deploy-pipeline` | Provision + deploy to Hetzner (idempotent) |
| `uv run remote-destroy` | Delete remote server |
| `uv run remote-run-pipeline <run>` | Run pipeline on remote in background |
| `uv run remote-run-cmd <cmd...>` | Run arbitrary command on remote |
| `uv run remote-run-bg <cmd...>` | Run command on remote in background |
| `uv run remote-bg-log [--follow] [N]` | Tail background job log |
| `uv run remote-bg-kill` | Kill background job |
| `uv run remote-download-store <path> <local>` | Download store files from remote |
| `uv run remote-ip` | Print remote server IP |
| `uv run remote-clifton-auth` | Refresh Clifton VPN certificate on remote |
