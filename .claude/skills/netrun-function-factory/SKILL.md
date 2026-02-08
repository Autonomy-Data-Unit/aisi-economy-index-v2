---
name: netrun-function-factory
description: "How the netrun function factory (netrun.node_factories.from_function) maps Python function signatures to node ports. Covers special parameters (ctx, print), multiple output ports (dict return), list[T] batch inputs, port groups (dot notation), _node_config attribute overrides, import path formats (dotted and file-path), and factory_args options."
---

# Nodes & the Function Factory

The most common way to create nodes is via the **function factory** (`netrun.node_factories.from_function`). It turns a regular Python function into a node by parsing its signature.

## Function Signature to Ports

- **Parameters** become **input ports** (one packet per parameter).
- **Return annotation** becomes **output port(s)**.

```python
def my_node(a: int, b: str) -> float:
    return float(a) + len(b)
# Input ports: a (int), b (str)
# Output port: out (float)
```

## Special Parameters

Two parameter names are reserved and do **not** become input ports:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx` | `NodeExecutionContext` | Access to execution context, packet operations, variables, retry info |
| `print` | callable | Captured print function — output is logged with timestamps |

```python
def my_node(data: str, print, ctx) -> str:
    print(f"Processing on attempt {ctx.retry_count + 1}")
    return data.upper()
```

## Multiple Output Ports

Return a dict annotation to create multiple output ports:

```python
def analyze(value: int, print) -> {"summary": str, "breakdown": str}:
    return {
        "summary": f"Result: {value}",
        "breakdown": f"{value} is {'even' if value % 2 == 0 else 'odd'}",
    }
# Output ports: summary (str), breakdown (str)
```

## List Input Ports

Annotate a parameter as `list[T]` to consume **all** packets at that port (instead of one):

```python
def batch_processor(data: list[str], print) -> str:
    print(f"Processing batch of {len(data)} items")
    return ", ".join(data)
```

## Port Groups (Dot Notation)

Use dots in output port names to create collapsible groups in the UI:

```python
def extract_features(item: str) -> {"features.color": str, "features.shape": str, "features.size": str}:
    return {
        "features.color": "red",
        "features.shape": "circle",
        "features.size": "large",
    }
```

## `_node_config` Attribute

Attach a `_node_config` attribute to a function to merge additional configuration. Accepts a `NodeConfig`, dict, or TOML string:

```python
def format_result(value: int) -> str:
    return f"The answer is: {value}"

format_result._node_config = '''
[extra]
description = "Formats the final result"
category = "output"
'''
```

## Import Path Formats

The `func` argument in `factory_args` supports two formats:

| Format | Example | Description |
|--------|---------|-------------|
| Dotted path | `"nodes.my_func"` | Standard Python import path |
| File path | `"./nodes.py::my_func"` | Relative to `project_root`, `::` separates file from attribute |

```json
{"factory_args": {"func": "nodes.double"}}
{"factory_args": {"func": "./nodes.py::double"}}
{"factory_args": {"func": "../shared/utils.py::helper"}}
```

## Factory Args in Configuration

```json
{
  "name": "my_node",
  "factory": "netrun.node_factories.from_function",
  "factory_args": {
    "func": "nodes.my_func",
    "include_port_types": true,
    "manual_output": false
  },
  "execution_config": {
    "pools": ["main"],
    "type_checking_enabled": true
  }
}
```

- `include_port_types` (default `true`) — Include type annotations as port types for validation.
- `manual_output` (default `false`) — When `true`, the function must manage output via `ctx` directly and return `None`.

## Sample Projects

- **00** (`sample_projects/00_basic_net_project/`): Basic function factory, single/multi-output, `_node_config` attribute
- **03** (`sample_projects/03_subgraphs/`): Port groups (dot notation), multi-output
- **05** (`sample_projects/05_advanced_flow_control/`): Batch `list[T]` inputs, multi-output
- **06** (`sample_projects/06_actions_and_recipes/`): File-path import format (`./nodes.py::func`)
