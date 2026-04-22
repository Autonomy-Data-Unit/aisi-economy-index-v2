# CLAUDE.md

## Project Overview

**ai-index** (AISI Exposure Index) is a productionized data pipeline for analyzing AI exposure in the economy. It matches job advertisements to O\*NET occupations and computes AI impact metrics (ASPECTT vectors, AI exposure scores, seniority/job zone).

## Pipeline DAG

The pipeline is defined in `config/netrun.json`. Run `uv run netrun validate -c config/netrun.json` to check the current node/edge counts. Each node is a module at `ai_index.nodes.<name>` (developed as `pts/ai_index/nodes/<name>.pct.py`). The pipeline has three parallel tracks that converge at index construction:

### Stage 1: Data Ingestion & Preparation

Two independent startup branches run in parallel:

- **`fetch_adzuna`** -- Downloads Adzuna job ad data from S3, deduplicates, stores in DuckDB.
- **`fetch_onet`** -- Downloads and extracts the O\*NET database.
- **`sample_ads`** -- Triggered after `fetch_adzuna`. Samples N ad IDs (or passes all through if `sample_n == -1`).
- **`prepare_onet_targets`** -- Triggered after `fetch_onet`. Filters O\*NET occupations, builds rich text descriptions combining titles, tasks, skills, and work activities. Writes `onet_targets.parquet`.

### Stage 2a: Job Ad Matching (blue path)

Matches job ads to O\*NET occupations through a multi-stage retrieval pipeline:

1. **`embed_ads`** -- Embeds raw job ad text (title + description) using the configured embedding model. Processes in chunks, stores embeddings in DuckDB.
2. **`embed_onet`** -- Embeds O\*NET occupation descriptions. Triggered after `prepare_onet_targets` via `broadcast_onet_ready`.
3. **`cosine_candidates`** -- Computes cosine similarity between ad and O\*NET embeddings. Selects top-k candidates per ad (default 20). Writes candidates parquet.
4. **`llm_filter_candidates`** -- LLM-based negative selection. Presents candidates to an LLM which selects functional matches. Supports structured output, unstructured, and reasoning model prompt variants. Writes filtered matches to DuckDB.
5. **`rerank_candidates`** -- Cross-encoder reranking of filtered candidates using Qwen3-Reranker. Produces final match scores.

### Stage 2b: O\*NET Exposure Scoring (green path)

Runs in parallel with job ad matching, triggered by `broadcast_onet_ready` (1-to-4 fan-out):

- **`score_presence`** -- Computes humanness/presence scores (physical, emotional, creative dimensions) per occupation from O\*NET work context, GWAs, and skills data.
- **`score_felten`** -- Computes Felten AIOE ability-application AI exposure scores per occupation, with configurable progress scenarios.
- **`score_task_exposure`** -- LLM-based task-level AI exposure classification. Evaluates each O\*NET task for AI automation potential and aggregates to occupation level.
- **`join_scores`** -- Synchronization barrier (3-to-1 join). Waits for all three score nodes to complete.
- **`combine_onet_exposure`** -- Merges all score DataFrames into a single combined exposure table. Validates occupation set consistency.

### Stage 3: Index Construction (red path)

Converges the matching and scoring results:

- **`compute_job_ad_exposure`** -- Maps occupation-level exposure scores to individual job ads via rerank-score-weighted averaging.
- **`aggregate_geo`** -- Aggregates ad-level AI exposure scores by Local Authority District (LAD22CD). Produces the final geographic index.

### Infrastructure Nodes

- **`broadcast_onet_ready`** -- Fan-out (1-to-4): triggers `embed_onet`, `score_presence`, `score_felten`, `score_task_exposure` after O\*NET preparation completes.
- **`join_scores`** -- Synchronization barrier (3-to-1): collects outputs from the three score nodes before combining.

### Node Storage Convention

Three storage tiers: `store/inputs/` (run-independent source data), `store/pipeline/{run_name}/` (run-specific intermediates), `store/outputs/` (final outputs). Storage format by use case: DuckDB via `ResultStore` for incremental/resumable writes (LLM nodes), Parquet for tabular outputs, NumPy for dense arrays (embeddings), CSV for small human-readable outputs.

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
- **pyinfra** - Remote server configuration (used by deploy module)
- **hcloud** CLI - Hetzner Cloud server provisioning (external, not a Python dep)

## Running the Pipeline

The pipeline is run via `run_pipeline_async(run_name)` (or the `run-pipeline` CLI entry point). The flow:

1. Load `.env` via `dotenv`
2. Load `run_defs.toml` — `_load_run_defs()` parses the TOML file
3. Resolve run definition — `_resolve_run_defs(run_defs, run_name)` merges `[defaults]` with `[runs.<run_name>]`, producing `(global_node_vars, per_node_vars)` dicts. Scalar values become global node vars; subtable dicts become per-node overrides.
4. Load netrun config — `NetConfig.from_file(netrun.json, global_node_vars=..., node_vars=...)` injects the resolved values into the graph's unfilled `NodeVariable` placeholders
5. Execute — `async with Net(config) as net:` starts the net, then loops `run_until_blocked()` until no progress

Run name is determined by: explicit argument > `RUN_NAME` env var > `"baseline"`.

### Key files
- `src/ai_index/run_pipeline.py` — `run_pipeline_async()`, `_load_run_defs()`, `_resolve_run_defs()`
- `src/ai_index/const.py` — Path constants (`store_path`, `inputs_path`, `outputs_path`, `onet_exposure_scores_path`, `aspectt_vectors_path`, config paths)
- `config/run_defs.toml` — Run definitions
- `config/netrun.json` — Pipeline graph with unfilled node_var placeholders

## Configuration Files

### `run_defs.toml` — Run definitions

Defines named pipeline configurations. `[defaults]` provides base values; `[runs.<name>]` overrides them. Scalar values become global node_vars; subtable dicts (e.g. `[defaults.llm_summarise]`) become per-node overrides. All values are accessible via `ctx.vars` in nodes.

**Convention:** Default values for all node variables (both global and per-node) live in `run_defs.toml`, not in `netrun.json`. The `netrun.json` only declares variable names and types as unfilled placeholders.

### `embed_models.toml` / `llm_models.toml` / `rerank_models.toml` — Model configs

Model-key-based lookup. Each model entry has a `mode` (api/local/sbatch) and model-specific params. Resolution: `_load_model_config(config_path, model_key)` looks up `models.<key>`, reads `mode`, merges `defaults.<mode>` with the model entry, returns `(mode, merged_dict)`.

#### Embedding model prompt/instruction support

Embedding models have three categories of prompt support, configured in `embed_models.toml`:

**1. Fixed prefixes** (`query_prefix` / `document_prefix`): Strings that must always be prepended to inputs for the model to work correctly. These are model-specific and unconditional. Example: e5-large requires `query_prefix = "query: "` and `document_prefix = "passage: "`.

**2. Named prompts** (`query_prompt_name` / `document_prompt_name`): Reference a named prompt from the model's SentenceTransformer config. Applied unconditionally. Example: `query_prompt_name = "query"` for arctic-embed-l-v2.

**3. Custom task instructions** (`supports_prompt = true`): The model accepts a free-form task instruction via the `prompt` parameter of `embed()`. The actual instruction text is defined in the pipeline node (e.g. via `run_defs.toml`), not in the model config. Only instruction-following models support this (Qwen3-Embedding, llama-embed-nemotron).

Pipeline nodes should:
- Always apply `query_prefix`/`document_prefix` if present in the model config
- Always apply `query_prompt_name`/`document_prompt_name` if present
- Only pass `prompt` if `supports_prompt = true` in the model config
- The `prompt` text itself comes from node configuration (e.g. a node var)

### `deploy.toml` — Remote deployment config

Settings for provisioning and managing a Hetzner Cloud server. Sections: `[server]` (name, type, location, image, SSH key), `[repo]` (remote path).

### `netrun.json` — Pipeline graph

Defines the DAG, node_var placeholders, and cache settings. All node_vars (global and per-node) are declared with types only — no default values. Defaults live in `run_defs.toml`.

Key global node_vars: `sample_n`, `sample_seed`, `embedding_model`, `llm_model`, `cosine_mode`, `topk`, `llm_batch_size`, `llm_max_new_tokens`, `run_name`, `adzuna_s3_prefix` (from `$env`).

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
```

### Key files
- `src/ai_index/utils/` — `embed()`, `llm_generate()`, `cosine_topk()`, `_load_model_config()`, `_resolve_model_args()`
- `config/embed_models.toml` — Embedding model configs
- `config/llm_models.toml` — LLM model configs

## `ai_index.utils` — Pipeline utilities

Model-key-based utility functions in `src/ai_index/utils/`. Each function resolves its execution mode from the TOML config, so callers only pass a model key. All have sync and async variants (e.g. `embed`/`aembed`).

- **`embed(texts, *, model, **kwargs) -> np.ndarray`** — Embed texts. Routes by mode (api/local/sbatch).
- **`llm_generate(prompts, *, model, **kwargs) -> list[str]`** — Generate LLM responses. Routes by mode. All three backends support `system_message`, `max_new_tokens`, and `json_schema` kwargs.
- **`cosine_topk(A, B, k, *, mode, **kwargs) -> dict`** — Top-K cosine similarity. Returns `{"indices": (n,k), "scores": (n,k)}`.

Explicit `**kwargs` override TOML config values. All functions support an optional `slurm_accounting={}` kwarg for Slurm resource tracking in sbatch mode.

**`OnetScoreSet`** (`scoring.py`): Standard output for score nodes. Validated DataFrame with `onet_code` + float score columns in [0, 1]. Provides `.validate()` and `.save(output_dir)`.

## Isambard HPC

The `isambard_utils` package (`src/isambard_utils/`) automates GPU workloads on the Isambard AI Phase 2 cluster (NVIDIA GH200 120GB, ARM64, Slurm). It handles SSH, file transfer, environment bootstrap, SBATCH job submission/polling, HuggingFace model caching, and Slurm accounting. Config: `config/isambard.toml` + `ISAMBARD_HOST` env var.

The high-level entry point is `orchestrate.arun_remote()`, which manages the full lifecycle: setup, model caching, input transfer, SBATCH submit, poll, accounting collection, output download, and cleanup. Billing: 0.25 NHR per GPU-hour for typical 1-GPU jobs.

Integration tests: `pytest src/tests/isambard_utils/` (requires active Clifton cert).

**Clifton VPN certificates:** The user has access to Clifton certificates for Isambard. These expire every ~12 hours and may need to be re-certified. If sbatch commands fail with SSH/connection errors, check whether the certificate has expired. See `CLAUDE.local.md` for auto-renewal procedure.

### Staged Data System

Instead of uploading serialized data per sbatch job (500MB each), raw data files (parquet, JSON) are staged on Isambard once and each job reads its chunk directly from Lustre.

- `isambard_utils/staging.py` -- `StagedRef`, `astage_file()`, `astage_files()` for idempotent uploads to `.staged_data/{content_hash}/`
- `isambard_utils/orchestrate.py` -- `StagedInput` dataclass (resolver name + source refs + chunk params). Hashed instantly from source hashes + params. `_download_outputs()` is throttled by a semaphore (`max_concurrent_downloads`, default 10) to prevent OOM.
- `llm_runner/staged.py` -- Resolver registry (`@resolver()` decorator). Resolvers run on the GPU node, read staged parquets with predicate pushdown, and build model inputs locally.

**Determinism:** Files used in content hashing must produce identical bytes across pipeline restarts. Use `ORDER BY` in DuckDB exports and `sort_keys=True` in JSON dumps. Use skip-if-exists for expensive exports (sample_ads/ad_texts.parquet, llm_filter/filtered_matches.parquet) to avoid re-writing files that haven't changed.

**Memory:** For large lookup structures (e.g. 38M candidate rows), use pyarrow tables with offset indexes rather than Python dicts. A sorted Arrow table with a `dict[ad_id, (start, end)]` index uses ~500MB vs ~15GB for equivalent Python dicts.

## Project Structure

```
├── nblite.toml              # nblite config (export pipelines)
├── pyproject.toml            # Package config (ai-index)
├── .env                      # Environment variables (ISAMBARD_HOST, ADZUNA_S3_PREFIX, etc.)
├── config/                   # All configuration files
│   ├── netrun.json           # Pipeline graph with node_var placeholders
│   ├── run_defs.toml         # Run definitions (defaults + named runs)
│   ├── embed_models.toml     # Embedding model configs
│   ├── llm_models.toml       # LLM model configs
│   ├── deploy.toml           # Remote deployment config (Hetzner server, storage box)
│   └── prompt_library/       # Prompt templates (Markdown files: llm_filter, score_task_exposure)
├── agent-context/            # Reference docs for netrun & nblite
│
│   ## nblite-managed (edit pts/, export to nbs/ and src/)
├── pts/ai_index/nodes/       # Node notebooks (.pct.py) - EDIT THESE
├── nbs/ai_index/nodes/       # Node notebooks (.ipynb, auto-generated from pts)
├── pts/scratch/              # Scratch/experiment notebooks (.pct.py) - EDIT THESE
├── nbs/scratch/              # Scratch notebooks (.ipynb, auto-generated from pts)
├── pts/validation/           # Validation notebooks (.pct.py) - EDIT THESE
├── nbs/validation/           # Validation notebooks (.ipynb, auto-generated from pts)
│
│   ## Plain Python (edit src/ directly)
├── src/ai_index/             # Pipeline package
│   ├── const.py              # Path constants
│   ├── run_pipeline.py       # Pipeline runner
│   ├── utils/                # embed(), llm_generate(), cosine_topk(), etc.
│   └── nodes/                # Node modules (auto-generated from pts/ai_index/nodes/)
├── src/isambard_utils/       # Isambard HPC utils
│   └── assets/
│       └── config.toml       # Isambard cluster config
├── src/llm_runner/           # LLM runner (embed, cosine, LLM generate)
├── src/dev_utils/            # Development utilities (set_node_func_args, etc.)
├── src/calibration/          # GPU-hours calibration tools
├── src/deploy/               # Remote deployment tools
├── src/validation/           # Validation modules (auto-generated from pts/validation/)
├── src/tests/                # Test modules
│
├── scripts/deploy_setup.py   # Standalone pyinfra deploy script (server setup)
└── .claude/skills/           # Netrun skill docs for Claude
```

## Development Workflow

### Where to edit code

Only three code locations use the nblite export pipeline (pts/ <-> nbs/ <-> src/):
- **`pts/ai_index/nodes/`** — Pipeline node notebooks. Edit `.pct.py` here, then run `nbl export --reverse && nbl export`.
- **`pts/scratch/`** — Scratch/experiment notebooks. Syncs pts <-> nbs only (no lib export).
- **`pts/validation/`** — Validation notebooks. Edit `.pct.py` here, then run `nbl export --reverse && nbl export`.

Everything else lives directly in `src/` and is edited there:
- `src/ai_index/utils/`, `src/ai_index/const.py`, `src/ai_index/run_pipeline.py`
- `src/isambard_utils/`, `src/llm_runner/`, `src/dev_utils/`
- `src/calibration/`, `src/deploy/`, `src/tests/`

**Exception: `__init__.py` files** — nblite skips dunder-named files during module export. For nblite-managed locations, edit `__init__.py` directly in `src/` and keep in sync with the corresponding `pts/.../__init__.pct.py` notebook.

### nblite commands
```bash
nbl export                    # Export nbs -> pts -> src
nbl export --reverse          # Sync pts changes back to nbs
nbl test                      # Test notebooks execute without errors
nbl fill                      # Execute notebooks and save outputs
nbl new pts/ai_index/nodes/foo.pct.py  # Create new node notebook
nbl new --template dev/templates/func_node.pct.py.jinja pts/ai_index/nodes/foo.pct.py  # Create new node from template
```

### Export pipeline (from nblite.toml)
```
nbs_nodes -> lib_nodes          (nbs/ai_index/nodes/ -> src/ai_index/nodes/)
nbs_nodes -> pts_nodes          (nbs/ai_index/nodes/ -> pts/ai_index/nodes/)
nbs_scratch -> pts_scratch      (nbs/scratch/ -> pts/scratch/, no lib)
nbs_validation -> lib_validation (nbs/validation/ -> src/validation/)
nbs_validation -> pts_validation (nbs/validation/ -> pts/validation/)
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

## Top-level await in notebooks

Notebooks (`.ipynb` / `.pct.py` via nblite) run in an async context. You can use `await` directly at the top level without wrapping in `asyncio.run()`. In fact, `asyncio.run()` will **error** in a notebook because an event loop is already running. Always use bare `await` for async calls in notebook cells.

```python
# WRONG — RuntimeError: cannot run event loop while another loop is running
result = asyncio.run(some_async_func())

# RIGHT — top-level await works in notebooks
result = await some_async_func()
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

## GPU-Hours Calibration

The `src/calibration/` module contains tools for measuring per-ad GPU timing and estimating costs for full pipeline runs on Isambard. Results are stored in `store/calibration/results/` (gitignored, regeneratable).

### Running calibration
```bash
uv run run-calibration <llm_model_key> <embedding_model_key>
# Example: uv run run-calibration qwen-7b-sbatch bge-large-sbatch
```

This runs the pipeline with the `[runs.calibration]` definition from `config/run_defs.toml` (`sample_n=1000`, shorter sbatch times, `resume=false` for LLM nodes). Cleans `store/pipeline/calibration/` before each run. The LLM and embedding model keys are injected dynamically. Results written to `store/calibration/results/{llm,embed}/`.

### Estimating GPU-hours
```bash
uv run estimate-calibration [N_ADS]  # default: 30,000,000
```

Reads all result JSONs and prints estimated hours, node-hours (NHR), and per-ad cost. When Slurm accounting data is available (from `sacct`), uses actual GPU execution time. Falls back to wall-clock time (which includes transfer overhead).

### `sbatch_cache` node variable

Global node var (`config/netrun.json`) inherited by all 6 sbatch-capable nodes. Default `true` in `config/run_defs.toml`. When `false`, forces fresh GPU execution by bypassing the content-addressed remote cache.

### `sbatch_time` node variable (IMPORTANT)

**Per-node** var (not global) on all 6 sbatch-capable nodes (`llm_summarise`, `llm_filter_candidates`, `embed_ads`, `embed_onet`, `cosine_match`, `score_task_exposure`). Controls the Slurm `--time` walltime limit for sbatch jobs submitted by that node.

**This value must be adjusted when changing `sample_n`.** The walltime needed scales with input size: 1000 ads needs minutes, 30M ads needs hours. If the walltime is too short, Slurm kills the job mid-execution. Defaults in `config/run_defs.toml` are set for full-scale runs (~30M ads). The `[runs.calibration]` section overrides them with shorter values suitable for `sample_n=1000`.

Nodes pass `time=sbatch_time` as a kwarg to `embed()`/`llm_generate()`/`cosine_topk()`, which overrides any `time` in the model TOML. In api/local modes, `time` is harmlessly stripped by `_strip_remote_kwargs()`.

### Key files
- `src/calibration/run_calibration.py` — CLI (`run-calibration`): run pipeline, collect timing, save results
- `src/calibration/calibrate_all.py` — CLI (`calibrate-all`): run all uncalibrated sbatch models
- `src/calibration/estimate.py` — CLI (`estimate-calibration`): read results, print GPU-hour estimates
- `store/calibration/results/{llm,embed}/*.json` — Per-model timing results (gitignored)

## Remote Deployment (Hetzner)

The `src/deploy/` module provisions a Hetzner Cloud server and deploys the pipeline for remote execution. The `store/` directory is created on the server's local disk. Configuration lives in `config/deploy.toml`. Requires `hcloud` CLI to be installed and authenticated.

### CLI commands
```bash
uv run remote-deploy-pipeline                          # Provision + deploy (idempotent)
uv run remote-destroy                                  # Delete server
uv run remote-run-cmd <command...>                     # Run command on remote (streams output)
uv run remote-run-bg <command...>                      # Run command in background (detached from SSH)
uv run remote-run-pipeline <run_name>                  # Shortcut: remote-run-bg run-pipeline <run_name>
uv run remote-bg-log [--follow] [N]                    # Tail background job log
uv run remote-bg-kill                                  # Kill background job
uv run remote-download-store <rel_path> <local_path>   # rsync store files to local
uv run remote-ip                                       # Print server IP
```

### Deploy flow (`remote-deploy-pipeline`)
1. Ensure SSH key registered in hcloud
2. Create server if not exists (hcloud)
3. Wait for SSH
4. Run pyinfra setup (`scripts/deploy_setup.py`): install system packages, uv
5. rsync code to remote (excludes `store/`, `.venv/`, `.env`)
6. Create `store` directory on local disk
7. Run `uv sync` on remote

Re-running is idempotent. If the server already exists, it skips provisioning and re-syncs code.

### Key files
- `config/deploy.toml` -- server and repo settings
- `src/deploy/deploy_setup.py` -- pyinfra deploy script (server setup)
- `src/deploy/config.py` -- config loading, hcloud helpers, SSH utilities
- `src/deploy/deploy_pipeline.py` -- main deploy orchestration
- `src/deploy/destroy.py`, `run_cmd.py`, `download_store.py`, `get_ip.py` -- other CLI commands
