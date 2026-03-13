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

Each node is a module at `ai_index.nodes.<name>` (developed as `pts/ai_index/nodes/<name>.pct.py`).

**Data ingestion & preparation:** `fetch_onet`, `fetch_adzuna`, `sample_ads`, `prepare_onet_targets`
**Job ad processing (matching):** `llm_summarise`, `embed_ads`, `embed_onet`, `cosine_match`, `llm_filter_candidates`
**O\*NET exposure scoring:** `build_aspectt_vectors`, `score_presence`, `score_felten`, `score_task_exposure`, `combine_onet_exposure`
**Index construction:** `compute_job_ad_exposure`, `aggregate_geo`, `compute_job_ad_aspectt_vectors` (disabled)
**Infrastructure:** `broadcast_onet_ready` (1->5), `broadcast_filter_done` (1->2), `join_scores` (3->1)

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

### `embed_models.toml` / `llm_models.toml` — Model configs

Model-key-based lookup. Each model entry has a `mode` (api/local/sbatch) and model-specific params. Resolution: `_load_model_config(config_path, model_key)` looks up `models.<key>`, reads `mode`, merges `defaults.<mode>` with the model entry, returns `(mode, merged_dict)`.

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
```

### Key files
- `pts/ai_index/utils/` — `embed()`, `llm_generate()`, `cosine_topk()`, `_load_model_config()`, `_resolve_model_args()`
- `config/embed_models.toml` — Embedding model configs
- `config/llm_models.toml` — LLM model configs

## `ai_index.utils` — Pipeline utilities

Model-key-based utility functions in `pts/ai_index/utils/`. Each function resolves its execution mode from the TOML config, so callers only pass a model key. All have sync and async variants (e.g. `embed`/`aembed`).

- **`embed(texts, *, model, **kwargs) -> np.ndarray`** — Embed texts. Routes by mode (api/local/sbatch).
- **`llm_generate(prompts, *, model, **kwargs) -> list[str]`** — Generate LLM responses. Routes by mode. All three backends support `system_message`, `max_new_tokens`, and `json_schema` kwargs.
- **`cosine_topk(A, B, k, *, mode, **kwargs) -> dict`** — Top-K cosine similarity. Returns `{"indices": (n,k), "scores": (n,k)}`.

Explicit `**kwargs` override TOML config values. All functions support an optional `slurm_accounting={}` kwarg for Slurm resource tracking in sbatch mode.

**`OnetScoreSet`** (`scoring.py`): Standard output for score nodes. Validated DataFrame with `onet_code` + float score columns in [0, 1]. Provides `.validate()` and `.save(output_dir)`.

## Isambard HPC

The `isambard_utils` package (`pts/isambard_utils/`) automates GPU workloads on the Isambard AI Phase 2 cluster (NVIDIA GH200 120GB, ARM64, Slurm). It handles SSH, file transfer, environment bootstrap, SBATCH job submission/polling, HuggingFace model caching, and Slurm accounting. Config: `src/isambard_utils/assets/config.toml` + `ISAMBARD_HOST` env var.

The high-level entry point is `orchestrate.arun_remote()`, which manages the full lifecycle: setup, model caching, input transfer, SBATCH submit, poll, accounting collection, output download, and cleanup. Billing: 0.25 NHR per GPU-hour for typical 1-GPU jobs.

Integration tests: `pytest src/tests/isambard_utils/` (requires active Clifton cert).

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
├── pts/calibration/          # GPU-hours calibration tools (.pct.py) - EDIT THESE
├── nbs/calibration/          # Calibration notebooks (auto-generated)
├── src/calibration/          # Calibration Python modules (auto-generated)
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

## GPU-Hours Calibration

The `pts/calibration/` module contains tools for measuring per-ad GPU timing and estimating costs for full pipeline runs on Isambard. Results are stored in `store/calibration/results/` (gitignored, regeneratable).

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
- `pts/calibration/run_calibration.pct.py` — CLI (`run-calibration`): run pipeline, collect timing, save results
- `pts/calibration/calibrate_all.pct.py` — CLI (`calibrate-all`): run all uncalibrated sbatch models
- `pts/calibration/estimate.pct.py` — CLI (`estimate-calibration`): read results, print GPU-hour estimates
- `store/calibration/results/{llm,embed}/*.json` — Per-model timing results (gitignored)

## Old Repository Reference

The old codebase is at `/Users/lukas/dev/20260208_e22t36__aisi-economy-index` (manually-run notebooks, now fully rebuilt in v2).
