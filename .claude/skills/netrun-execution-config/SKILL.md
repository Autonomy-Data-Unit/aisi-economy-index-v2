---
name: netrun-execution-config
description: "Complete netrun NodeExecutionConfig field reference: pools, retries, retry_wait, timeout, max_epochs, max_parallel_epochs, rate_limit_per_second, capture_prints, print_flush_interval, print_echo_stdout, type_checking_enabled, propagate_exceptions, defer_startup, on_node_failure, cache. Also covers node variables (node_vars): global and per-node definitions, variable types (str/int/float/bool/json), and ctx.vars access."
---

# Execution Configuration

Each node can have an `execution_config` section controlling how it runs.

## Full Example

```json
{
  "name": "my_node",
  "factory": "netrun.node_factories.from_function",
  "factory_args": {"func": "nodes.my_func"},
  "execution_config": {
    "pools": ["main"],
    "retries": 3,
    "retry_wait": 0.5,
    "timeout": 10.0,
    "max_epochs": 100,
    "max_parallel_epochs": 4,
    "rate_limit_per_second": 10.0,
    "capture_prints": true,
    "print_flush_interval": 0.1,
    "print_echo_stdout": false,
    "type_checking_enabled": true,
    "propagate_exceptions": true,
    "on_node_failure": "nodes.on_failure"
  }
}
```

## Complete Field Reference

| Field | Type | Default | Description |
|-------|---------|---------|-------------|
| `pools` | `list[str]` | `["main"]` | Which pool(s) can execute this node |
| `retries` | `int` | `0` | Number of retry attempts on failure |
| `retry_wait` | `float` | `0.0` | Seconds to wait between retries |
| `timeout` | `float` | `null` | Max seconds for execution (requires thread/process pool) |
| `max_epochs` | `int` | `null` | Total epoch limit for this node |
| `max_parallel_epochs` | `int` | `null` | Max concurrent epochs |
| `rate_limit_per_second` | `float` | `null` | Max epochs per second |
| `capture_prints` | `bool` | `true` | Capture print output with timestamps |
| `print_flush_interval` | `float` | `0.1` | How often to flush print buffer (seconds) |
| `print_buffer_max_size` | `int` | `null` | Max print buffer entries |
| `print_echo_stdout` | `bool` | `false` | Also print to real stdout |
| `type_checking_enabled` | `bool` | `null` | Override net-level type checking |
| `propagate_exceptions` | `bool` | `null` | Override net-level exception propagation |
| `print_exceptions` | `bool` | `null` | Override net-level exception printing |
| `defer_startup` | `bool` | `false` | Defer `start_node_func` until first epoch |
| `pool_allocation_method` | `str` | `null` | Per-node pool allocation override |
| `node_vars` | `dict[str, NodeVariable]` | `null` | Per-node variable overrides |
| `on_node_failure` | `str` | `null` | Failure callback (import path or callable) |
| `exec_node_func` | `str` | `null` | Custom execution function (import path) |
| `start_node_func` | `str` | `null` | Called on pool start (import path) |
| `stop_node_func` | `str` | `null` | Called on pool stop (import path) |
| `cache` | `NodeCacheConfig` | `null` | Per-node cache overrides |

## Net-Level Defaults

Some settings can be set at the net config level as defaults for all nodes:

```json
{
  "type_checking_enabled": true,
  "propagate_exceptions": true,
  "print_exceptions": false,
  "default_pool_allocation_method": "round-robin",
  "dead_letter_queue": true,
  "graph": { ... }
}
```

Per-node `execution_config` values override these net-level defaults.

---

# Node Variables

Node variables provide key-value configuration accessible inside node functions via `ctx.vars`.

## Global Variables

Defined at the net config level, available to all nodes:

```json
{
  "node_vars": {
    "label":   {"value": "primes", "type": "str"},
    "verbose": {"value": "false",  "type": "bool"}
  }
}
```

TOML equivalent:

```toml
[node_vars.label]
value = "primes"
type = "str"

[node_vars.verbose]
value = "false"
type = "bool"
```

## Per-Node Overrides

Override global values for a specific node:

```json
{
  "execution_config": {
    "node_vars": {
      "label":   {"value": "process-worker", "type": "str"},
      "verbose": {"value": "true", "type": "bool"}
    }
  }
}
```

## Variable Types

| Type | Example Value | Resolved Python Type |
|------|---------------|---------------------|
| `str` | `"hello"` | `str` |
| `int` | `"42"` | `int` |
| `float` | `"3.14"` | `float` |
| `bool` | `"true"` / `"false"` | `bool` |
| `json` | `'{"key": "value"}'` | parsed via `json.loads` |

Note: Values are always stored as strings and resolved to the target type at runtime.

## Accessing in Node Functions

```python
def find_primes(n: int, ctx, print) -> list:
    label = ctx.vars.get("label", "default")
    verbose = ctx.vars.get("verbose", False)
    if verbose:
        print(f"[{label}] Finding primes up to {n}")
    return [x for x in range(2, n) if all(x % i for i in range(2, x))]
```

## Sample Projects

- **01** (`sample_projects/01_thread_and_process_pools/`): Node variables, `ctx.vars`, multiple pool types, `max_epochs`
- **04** (`sample_projects/04_error_handling/`): Retries, timeouts, `on_node_failure`, `propagate_exceptions`
- **05** (`sample_projects/05_advanced_flow_control/`): `rate_limit_per_second`
- **06** (`sample_projects/06_actions_and_recipes/`): Node variables in TOML format
