# CLAUDE.md

## Project Overview

**ai-index** (AISI Economy Index v2) is a productionized data pipeline for analyzing AI exposure in the economy. It matches job advertisements to O\*NET occupations and computes AI impact metrics (ASPECTT vectors, AI exposure scores, seniority/job zone).

This is a clean rewrite of the old repository at `/Users/lukas/dev/20260208_e22t36__aisi-economy-index`, which was a collection of manually-run notebooks. The v2 uses **netrun** for orchestrating the data pipeline and **nblite** for literate programming development.

## Pipeline DAG

The pipeline is defined in `config/netrun.json`. It currently contains 4 nodes and 2 edges. The matching, scoring, and analysis stages are being rebuilt (see `_dev/aisi_demo_pipeline_analysis.md` for the full plan).

```
  fetch_onet (run_on_startup)
     │
     │ onet_tables
     ▼
  (not yet connected to downstream)


  fetch_adzuna (run_on_startup)
     │ signal: epoch_finished
     ▼
  sample_ads
     │
     │ ad_ids
     ▼
  llm_summarise
     │
     │ summary_meta
     ▼
  (not yet connected to downstream)
```

### Nodes (4 total, no subgraphs)
- `fetch_onet` (run_on_startup) — Download and extract O\*NET 30.0 database. Output: `onet_tables`
- `fetch_adzuna` (run_on_startup) — Download raw Adzuna job ads from S3 to DuckDB, deduplicate. Signals `epoch_finished` to trigger `sample_ads`.
- `sample_ads` — Sample job ads for processing (or pass through all if `sample_n=-1`). Output: `ad_ids` (np.ndarray or None).
- `llm_summarise` — Run LLM to extract structured summaries from job ads using structured JSON output (`json_schema` parameter). Processes ads in configurable chunks with incremental parquet writes and resume support. Input: `ad_ids`. Output: `summary_meta` (dict with parquet path + counts). Writes summaries to `store/pipeline/{run_name}/summaries.parquet`. Node vars: `llm_batch_size`, `llm_max_new_tokens` (global), `summarise_resume` (per-node).

### Planned pipeline stages (not yet implemented)

The full pipeline will eventually include exposure scoring, job-ad-to-occupation matching, and index generation stages. These were present in an earlier iteration but have been removed for a rebuild. See the Old Repository Reference section for context on the intended pipeline.

### Node Function Paths

Each node is a module at `ai_index.nodes.<name>` (developed as `pts/ai_index/nodes/<name>.pct.py`).

### Old Pipeline Reference

The old pipeline (to be rebuilt) had these stages:
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
- `pts/ai_index/const.pct.py` — Path constants (`assets_path`, `store_path`, `inputs_path`, config paths)
- `src/ai_index/assets/run_defs.toml` — Run definitions
- `src/ai_index/assets/netrun.json` — Pipeline graph with unfilled node_var placeholders

## Configuration Files

### `run_defs.toml` — Run definitions

Defines named pipeline configurations. `[defaults]` provides base values; `[runs.<name>]` overrides them. Scalar values become global node_vars; subtable dicts (e.g. `[defaults.llm_summarise]`) become per-node overrides. All values are accessible via `ctx.vars` in nodes.

**Convention:** Default values for all node variables (both global and per-node) live in `run_defs.toml`, not in `netrun.json`. The `netrun.json` only declares variable names and types as unfilled placeholders.

```toml
[defaults]
years = ""               # Comma-delimited year filter ("" = all years)
sample_n = 0             # 0 = full run, N = sample N ads
sample_seed = 42
embedding_model = "text-embedding-3-large"   # Key into embed_models.toml
cosine_mode = "api"      # "api", "local", or "sbatch"
topk = 5                 # Top-K candidates for cosine similarity
llm_model = "gpt-5.2"   # Key into llm_models.toml
llm_batch_size = 1000    # Number of prompts per LLM call
llm_max_new_tokens = 220 # Max tokens per LLM response
llm_max_concurrent_batches = 1   # Max concurrent batch LLM calls

[defaults.fetch_adzuna]
fetch_years = "all"

[defaults.llm_summarise]
summarise_resume = true          # Resume from previous partial run
summarise_max_retries = 0        # Retry rounds for failed ads

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
dtype = "float16"
backend = "vllm"
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

Configures the Isambard AI Phase 2 cluster connection. Symlinked from `src/isambard_utils/assets/config.toml`.

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

Key global node_vars: `years`, `sample_n`, `sample_seed`, `embedding_model`, `llm_model`, `cosine_mode`, `topk`, `llm_batch_size`, `llm_max_new_tokens`, `llm_max_concurrent_batches`, `run_name`, `adzuna_s3_prefix` (from `$env`).

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
   - Access via `ctx.vars.get("my_var", default)` in the node function

**Convention:** `netrun.json` only declares names and types. All default values go in `run_defs.toml`.

All node vars are accessible in node functions via `ctx.vars["var_name"]`. Values from TOML are Python-typed (int, str, bool) but may need explicit casting with `int()` when the type system returns strings.

## Execution Modes

Execution mode is determined per-model via the `mode` field in `embed_models.toml` / `llm_models.toml`. The pipeline's `embedding_model`, `llm_model`, and `cosine_mode` node_vars select which model config (and therefore which mode) to use.

| Mode | Description |
|------|-------------|
| `api` | No GPU needed: embeddings via OpenAI/Gemini API, cosine sim via numpy, LLM via `adulib.llm` (litellm) |
| `local` | Direct CUDA on current machine (sentence-transformers, torch). Mac variants use CPU/float32. |
| `sbatch` | Orchestrate from local: serialize inputs, submit SBATCH job to Isambard, wait, download results |

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
├── isambard_config.toml      # Symlink -> src/isambard_utils/assets/config.toml
├── netrun.json               # Symlink -> src/ai_index/assets/netrun.json
├── run_defs.toml             # Symlink -> src/ai_index/assets/run_defs.toml
├── embed_models.toml         # Symlink -> src/ai_index/assets/embed_models.toml
├── llm_models.toml           # Symlink -> src/ai_index/assets/llm_models.toml
├── agent-context/            # Reference docs for netrun & nblite
├── pts/ai_index/             # Source of truth (.pct.py files) - EDIT THESE
│   ├── const.pct.py          # Path constants
│   ├── utils.pct.py          # embed(), llm_generate(), cosine_topk()
│   ├── run_pipeline.pct.py   # Pipeline runner
│   └── nodes/                # Node functions (4 nodes)
├── nbs/ai_index/             # Jupyter notebooks (auto-generated from pts)
├── src/ai_index/             # Python modules (auto-generated) - DO NOT EDIT
│   └── assets/
│       ├── netrun.json       # Pipeline graph
│       ├── run_defs.toml     # Run definitions
│       ├── embed_models.toml # Embedding model configs
│       └── llm_models.toml   # LLM model configs
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
- `#|top_export` - Export at module level (for function-export notebooks)
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
