# netrun — Concise Reference

> **Audience:** This is a condensed reference for AI agents. For full explanations and extended examples, see `NETRUN_INSTRUCTIONS.md`. For working code, read the sample projects listed in the [index at the end](#sample-project-index).

## Concepts

**Nodes** have input/output **ports**. **Edges** connect output ports to input ports. Data flows as **packets**. When a node's **salvo condition** is met, an **epoch** (execution) fires, consuming input packets and producing output packets.

## Minimal Example

```python
# nodes.py
def double(x: int, print) -> int:
    print(f"Doubling {x}")
    return x * 2
```

```json
{
  "output_queues": {"results": {"ports": [["double", "out"]]}},
  "graph": {
    "nodes": [{"name": "double", "factory": "netrun.node_factories.from_function",
               "factory_args": {"func": "nodes.double"}}],
    "edges": []
  }
}
```

```python
# main.py
import asyncio
from netrun.core import Net, NetConfig

async def main():
    async with Net(NetConfig.from_file("main.netrun.json")) as net:
        net.inject_data("double", "x", [5])
        while True:
            await net.run_until_blocked()
            startable = net.get_startable_epochs()
            if not startable: break
            for eid in startable: await net.execute_epoch(eid)
        print(net.flush_output_queue("results"))  # [10]

asyncio.run(main())
```

## Configuration

- Files: `.netrun.json` or `.netrun.toml`. Load via `NetConfig.from_file(path)`.
- Convert: `netrun convert file.netrun.json -o file.netrun.toml`
- `project_root` defaults to the config file's directory; controls relative path resolution.
- **Env vars**: Any field accepts `{"$env": "VAR", "default": val}` (JSON) or `{"$env" = "VAR", default = val}` (TOML).

## Function Factory

`netrun.node_factories.from_function` creates nodes from Python functions:

- **Parameters** → input ports. **Return annotation** → output port(s).
- **Special params** (not ports): `ctx` (NodeExecutionContext), `print` (captured logger).
- **Multi-output**: `-> {"a": int, "b": str}` creates ports `a` and `b`.
- **Batch input**: `data: list[str]` consumes all packets at that port.
- **Port groups**: dot notation in keys (`"features.color"`) creates UI groups.
- **`_node_config`**: attach as attribute (TOML string, dict, or NodeConfig) for extra metadata.
- **Import formats**: `"nodes.func"` (dotted) or `"./nodes.py::func"` (file-path, relative to `project_root`).
- **Factory args**: `func` (required), `include_port_types` (default true), `manual_output` (default false).

## Graph

**Edges**: `{"source_str": "nodeA.port", "target_str": "nodeB.port"}`

**Subgraphs** (flattened at resolution, nodes prefixed with subgraph name):
- **Inline**: `{"type": "subgraph", "name": "...", "nodes": [...], "edges": [...], "exposed_in_ports": {...}, "exposed_out_ports": {...}}`
- **File-ref**: `{"type": "subgraph", "name": "...", "path": "./sub.netrun.json", "exposed_in_ports": {...}, ...}`
- **Factory**: `{"name": "...", "factory": "./factory.py", "factory_args": {...}}` where `get_node_config()` returns `SubgraphConfig`.

## Net Lifecycle & Execution

```python
async with Net(config) as net:          # start/stop via context manager
    net.inject_data("node", "port", [val])  # inject data
    # -- execution loop --
    while True:
        await net.run_until_blocked()
        startable = net.get_startable_epochs()
        if not startable: break
        for eid in startable: await net.execute_epoch(eid)
    # -- or use background mode --
    # await net.start_background(); await net.wait_until_done()
```

**Injection helpers**: `net.nodes["n"].inject("port", val)`, `.inject({"a": v1, "b": v2})`, `.inject({"a": [v1,v2]}, plural=True)`

**Output queues** (config: `"output_queues": {"name": {"ports": [["node","port"]]}}`):
- `net.flush_output_queue("name")` / `net.flush_all_output_queues()`
- `await net.get_output("name", timeout=5.0)` / `net.try_get_output("name")`
- `include_metadata=True` returns `ConsumedOutputPacket` with `.value`, `.from_node`, `.from_port`, `.timestamp`, `.epoch_id`.

## Execution Context (`ctx`)

| Property / Method | Description |
|---|---|
| `ctx.epoch_id`, `ctx.node_name` | Current epoch and node |
| `ctx.retry_count`, `ctx.retry_exceptions` | Retry state |
| `ctx.vars` | Resolved node variables dict |
| `ctx.create_packet(value)` → id | Create packet (manual mode) |
| `ctx.consume_packet(id)` → value | Consume packet (manual mode) |
| `ctx.load_output_port(port, id)` | Load packet to output (manual mode) |
| `ctx.send_output_salvo(name)` | Send output salvo (manual mode) |
| `ctx.cancel_epoch()` | Cancel — raises EpochCancelled, discards packets |
| `print(...)` | Captured with timestamp, appears in epoch logs |

## Execution Config (`execution_config`)

Key fields (set per-node, some overridable at net level):

| Field | Default | Notes |
|---|---|---|
| `pools` | `["main"]` | Which pool(s) run this node |
| `retries` / `retry_wait` | `0` / `0.0` | Retry attempts and delay |
| `timeout` | null | Requires thread/process pool |
| `max_epochs` | null | Total epoch limit |
| `max_parallel_epochs` | null | Concurrent epoch limit |
| `rate_limit_per_second` | null | Epoch rate limit |
| `type_checking_enabled` | null | Override net-level setting |
| `propagate_exceptions` | null | `false` → queue instead of raise |
| `on_node_failure` | null | Import path to callback (`NodeFailureContext`) |
| `capture_prints` / `print_echo_stdout` | true / false | Print capture settings |
| `cache` | null | Per-node `NodeCacheConfig` overrides |

**Net-level defaults**: `type_checking_enabled` (true), `propagate_exceptions` (true), `dead_letter_queue` (true), `default_pool_allocation_method` ("round-robin").

## Worker Pools

| Type | Config | Use Case |
|---|---|---|
| `main` | `{"type": "main"}` | Async, default |
| `thread` | `{"type": "thread", "num_workers": N}` | Blocking I/O, timeouts |
| `multiprocess` | `{"type": "multiprocess", "num_processes": N, "threads_per_process": M}` | CPU-bound |
| `remote` | `{"type": "remote", "url": "ws://...", "worker_name": "...", ...}` | Distributed |

Allocation methods: `round-robin` (default), `random`, `least-busy`.

## Node Variables

Global: `"node_vars": {"key": {"value": "val", "type": "str"}}` — types: `str`, `int`, `float`, `bool`, `json`.
Per-node override: same structure inside `execution_config.node_vars`.
Access: `ctx.vars["key"]`.

## Caching

Enable: `"cache": {"enabled": true, "include_all_nodes": true}` (or `include_nodes: ["pattern*"]`).
Modes: `both` (default, skip on hit), `output` (always run, record outputs), `input` (always run, record inputs).
Per-node: `execution_config.cache: {"enabled": true, "cache_what": "output", "version": 2}`.
Persistence: `"storage_path": "./.cache"` (default: temp dir, lost between runs).
API: `net.cache_stats()`, `net.get_cached_entries(node)`, `net.clear_cache()`, `net.clear_node_cache(node)`, `epoch.was_cache_hit`.

## Error Handling

- **Retries**: `retries: 3, retry_wait: 0.5` — access `ctx.retry_count` in node.
- **Failure callback**: `on_node_failure: "mod.func"` — receives `NodeFailureContext` with `.exception`, `.node_name`, `.retry_count`.
- **Cancel**: `ctx.cancel_epoch()` — discards in-flight packets.
- **Propagation**: `propagate_exceptions: false` → `net.exception_queue` / `net.propagate_exceptions()`.
- **DLQ**: `net.dead_letter_queue` / `net.clear_dead_letter_queue()`.
- **Timeout**: `timeout: 5.0` (requires thread/process pool).
- **Types**: `type_checking_enabled: true` → `PacketTypeMismatch` on mismatch.

## Salvo Conditions

Defaults auto-generated by function factory. Override for custom triggers:

```json
{"in_salvo_conditions": {"trigger": {
    "max_salvos": {"type": "finite", "max": 1},
    "ports": {"data": {"type": "count", "count": 3}},
    "term": {"type": "port", "port_name": "data",
             "state": {"type": "equals_or_greater_than", "value": 3}}
}}}
```

Port states: `empty`, `non_empty`, `full`, `non_full`, `equals`, `less_than`, `greater_than`, `equals_or_less_than`, `equals_or_greater_than`.
Terms: `true`, `false`, `port`, `and`, `or`, `not`.
Packet counts: `{"type": "all"}`, `{"type": "count", "count": N}`.

## Inspection

```python
node = net.nodes["name"]      # NodeInfo
node.in_port_names / .out_port_names / .epochs / .epoch_count / .is_busy
node.inject("port", val) / .inject_packets("port", [v1,v2])

net.print_all_logs()           # All logs
net.get_node_logs("name")      # [(timestamp, msg), ...]
net.epochs                     # {id: EpochRecord} — .node_name, .state, .was_cache_hit, .pool_id
net.edges                      # [EdgeInfo] — .source_node, .target_node, .packet_count
```

## Targeted Execution

```python
salvos = await net.run_to_targets("node_name")  # or ["n1", "n2"]
# Runs upstream, stops before target. Returns input salvos that would fire.
for s in salvos:
    for port, values in s.packets.items(): print(port, values)
```

## Actions & Recipes

**Actions**: shell commands in `extra.ui.actions` with template vars (`$NODE_NAME`, `$PROJECT_ROOT`, `$NET_FILE_PATH`, `$NET_FILE_DIR`, `$DEFAULT_CMD`, `$NODE_CONFIG`, plus custom `ui.env`).
**Recipes**: Python scripts (`run(config, inputs) -> config`) with optional `get_prompts(config)`. Prompt types: `text`, `number`, `select`, `checkbox`.

## CLI

| Command | Description |
|---|---|
| `netrun validate [-c CFG]` | Validate config |
| `netrun info [-c CFG]` | Summary stats |
| `netrun structure [-c CFG]` | Graph topology JSON |
| `netrun nodes [-c CFG]` | List nodes with ports |
| `netrun node NAME [-c CFG]` | Node detail |
| `netrun convert FILE [-o OUT]` | JSON ↔ TOML |
| `netrun factory-info PATH` | Factory params |
| `netrun actions list/run` | Manage actions |
| `netrun recipes list/run` | Manage recipes |

## netrun-ui

```bash
netrun-ui [FILE]           # Native window (background)
netrun-ui --fg             # Foreground
netrun-ui --server         # Browser mode
netrun-ui --dev            # Dev mode (Vite hot reload)
```

Options: `-p PORT`, `--frontend-port PORT`, `-C WORKDIR`, `--width W`, `--height H`.

---

## Sample Project Index

| # | Path | Key Features |
|---|------|---|
| 00 | `sample_projects/00_basic_net_project/` | Function factory, edges, output queues, multi-output, `_node_config`, print capture, logs |
| 01 | `sample_projects/01_thread_and_process_pools/` | Pool types (main/thread/multiprocess/remote), allocation methods, node vars, max_epochs, recipes |
| 02 | `sample_projects/02_remote_deployment/` | Remote pool server, cloud deployment, SSH tunneling |
| 03 | `sample_projects/03_subgraphs/` | Inline/file-ref/factory subgraphs, exposed ports, port groups (dot notation) |
| 04 | `sample_projects/04_error_handling/` | Retries, on_failure callback, cancel_epoch, type checking, propagate_exceptions, timeout, DLQ |
| 05 | `sample_projects/05_advanced_flow_control/` | Custom salvo conditions, finite slots, rate limiting, batch (list) inputs, multi-output |
| 06 | `sample_projects/06_actions_and_recipes/` | TOML format, file-path imports, actions, template vars, env vars, recipes, node vars |
| 07 | `sample_projects/07_run_to_targets/` | run_to_targets(), targeted testing, inspecting upstream data |
| 08 | `sample_projects/08_caching/` | Cache config, modes, version invalidation, persistent storage, cache API, NodeInfo helpers |

### Concept → Project Quick-Lookup

| Concept | Projects |
|---|---|
| Function factory, basic execution loop, output queues | 00 |
| Worker pools (thread/multiprocess/remote), allocation | 01, 02 |
| Subgraphs (inline, file-ref, factory), port groups | 03 |
| Retries, timeouts, cancel, DLQ, type checking | 04 |
| Custom salvos, finite slots, rate limiting, batch input | 05 |
| TOML config, file-path imports, actions, recipes | 06 |
| run_to_targets | 07 |
| Caching/memoization | 08 |
| Node variables (`ctx.vars`) | 01, 06 |
| `_node_config` attribute | 00 |
| Multi-output ports (`-> {"a": T}`) | 00, 03, 05 |
| Logging / print capture | 00, 01, 04, 07, 08 |
