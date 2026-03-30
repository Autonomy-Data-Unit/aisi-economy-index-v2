# NETRUN_INSTRUCTIONS.md

A practical guide to using **netrun**, a flow-based development runtime for Python.

> **Audience:** This document is primarily intended as context for AI agents working on netrun projects. For working code examples of every feature described here, see the [Sample Project Index](#20-sample-project-index) at the end — those projects are the authoritative source of real usage patterns and should be consulted whenever you need concrete implementation details.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Configuration](#3-configuration)
4. [Nodes & the Function Factory](#4-nodes--the-function-factory)
5. [Graph Topology](#5-graph-topology)
6. [Ports & Flow Control](#6-ports--flow-control)
7. [The Net Class](#7-the-net-class)
8. [Node Execution Context](#8-node-execution-context)
9. [Execution Configuration](#9-execution-configuration)
10. [Worker Pools](#10-worker-pools)
11. [Node Variables](#11-node-variables)
12. [Caching & Memoization](#12-caching--memoization)
13. [Error Handling](#13-error-handling)
14. [Inspecting Results](#14-inspecting-results)
15. [Targeted Execution](#15-targeted-execution)
16. [Actions & Recipes](#16-actions--recipes)
17. [CLI Reference](#17-cli-reference)
18. [netrun-ui](#18-netrun-ui)
19. [Configuration Reference Tables](#19-configuration-reference-tables)
20. [Sample Project Index](#20-sample-project-index)

---

## 1. Introduction

netrun is a flow-based development runtime where you define a **graph** of **nodes** connected by **edges**. Data flows through the graph as **packets**. Each node is a Python function (or custom implementation) that receives packets on its input ports and sends results out through its output ports.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Node** | A processing unit with input and output ports. Typically created from a Python function. |
| **Port** | A connection point on a node — either input or output. Has a slot capacity and optional type. |
| **Edge** | A directed connection from one node's output port to another node's input port. |
| **Packet** | A unit of data flowing through the network. Carries a value and has a unique ID. |
| **Epoch** | A single execution of a node. A node fires when its input salvo condition is met. |
| **Salvo** | A group of packets that enter or exit a node together. Salvo conditions define the trigger rules. |

### How It Works

1. Packets arrive at a node's input ports (via edges or injection).
2. When a salvo condition is satisfied, an epoch is created.
3. The node function executes, consuming input packets and producing output packets.
4. Output packets travel along edges to the next node's input ports.
5. The cycle repeats until no more work can be done.

---

## 2. Getting Started

### Installation

```bash
pip install netrun
# or
uv add netrun
```

### Project Structure

A typical netrun project:

```
my_project/
├── main.netrun.json   # Network configuration (or .toml)
├── nodes.py           # Node functions
└── main.py            # Execution script
```

### Minimal Example

**nodes.py** — Define node functions:

```python
def double(x: int, print) -> int:
    """Double the input value."""
    print(f"Doubling {x}")
    return x * 2

def add(a: int, b: int, print) -> int:
    """Add two numbers."""
    print(f"Adding {a} + {b}")
    return a + b
```

**main.netrun.json** — Define the graph:

```json
{
  "output_queues": {
    "results": {"ports": [["add", "out"]]}
  },
  "graph": {
    "nodes": [
      {
        "name": "double",
        "factory": "netrun.node_factories.from_function",
        "factory_args": {"func": "nodes.double"}
      },
      {
        "name": "add",
        "factory": "netrun.node_factories.from_function",
        "factory_args": {"func": "nodes.add"}
      }
    ],
    "edges": [
      {"source_str": "double.out", "target_str": "add.a"}
    ]
  }
}
```

**main.py** — Run the network:

```python
import asyncio
from pathlib import Path
from netrun.core import Net, NetConfig

async def main():
    config = NetConfig.from_file(Path(__file__).parent / "main.netrun.json")

    async with Net(config) as net:
        # Inject input data
        net.inject_data("double", "x", [5])
        net.inject_data("add", "b", [10])

        # Run until all processing is complete
        while True:
            await net.run_until_blocked()
            startable = net.get_startable_epochs()
            if not startable:
                break
            for epoch_id in startable:
                await net.execute_epoch(epoch_id)

        # Retrieve results
        results = net.flush_output_queue("results")
        for value in results:
            print(f"Result: {value}")  # Result: 20

asyncio.run(main())
```

The execution loop pattern is:
1. `run_until_blocked()` — moves packets along edges and creates startable epochs.
2. `get_startable_epochs()` — returns epochs ready to execute.
3. `execute_epoch()` — runs the node function for each epoch.
4. Repeat until no more startable epochs exist.

---

## 3. Configuration

### File Formats

netrun supports both JSON and TOML configuration files. Files must use the `.netrun.json` or `.netrun.toml` extension.

**JSON format:**

```json
{
  "output_queues": {
    "results": {"ports": [["my_node", "out"]]}
  },
  "graph": {
    "nodes": [
      {
        "name": "my_node",
        "factory": "netrun.node_factories.from_function",
        "factory_args": {"func": "nodes.my_func"}
      }
    ],
    "edges": []
  }
}
```

**Equivalent TOML format:**

```toml
[output_queues.results]
ports = [["my_node", "out"]]

[[graph.nodes]]
name = "my_node"
factory = "netrun.node_factories.from_function"
factory_args = { func = "nodes.my_func" }
```

### Loading Configuration

```python
from netrun.core import NetConfig

# From file
config = NetConfig.from_file("main.netrun.json")
config = NetConfig.from_file("main.netrun.toml")

# Or construct in code
config = NetConfig(graph=GraphConfig(nodes=[...], edges=[...]))
```

### Converting Between Formats

```bash
netrun convert main.netrun.json -o main.netrun.toml
netrun convert main.netrun.toml -o main.netrun.json
```

### Project Root

`project_root` controls how relative file paths are resolved (for imports, subgraph files, etc.). If not set explicitly, it defaults to the directory containing the config file.

```json
{
  "project_root": ".",
  "graph": { ... }
}
```

### Environment Variable Substitution

Any config field can reference an environment variable using `{"$env": "VAR_NAME"}` (JSON) or `{$env = "VAR_NAME"}` (TOML). An optional default is supported.

**JSON:**
```json
{
  "pools": {
    "workers": {
      "spec": {
        "type": "thread",
        "num_workers": {"$env": "NUM_WORKERS", "default": 4}
      }
    }
  }
}
```

**TOML:**
```toml
[pools.workers.spec]
type = "thread"
num_workers = { "$env" = "NUM_WORKERS", default = 4 }
```

Environment variables are resolved when the Net is created. If a variable is missing and no default is provided, an error is raised.

---

## 4. Nodes & the Function Factory

The most common way to create nodes is via the **function factory** (`netrun.node_factories.from_function`). It turns a regular Python function into a node by parsing its signature.

### Function Signature to Ports

- **Parameters** become **input ports** (one packet per parameter).
- **Return annotation** becomes **output port(s)**.

```python
def my_node(a: int, b: str) -> float:
    return float(a) + len(b)
# Input ports: a (int), b (str)
# Output port: out (float)
```

### Special Parameters

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

### Multiple Output Ports

Return a dict annotation to create multiple output ports:

```python
def analyze(value: int, print) -> {"summary": str, "breakdown": str}:
    return {
        "summary": f"Result: {value}",
        "breakdown": f"{value} is {'even' if value % 2 == 0 else 'odd'}",
    }
# Output ports: summary (str), breakdown (str)
```

### List Input Ports

Annotate a parameter as `list[T]` to consume **all** packets at that port (instead of one):

```python
def batch_processor(data: list[str], print) -> str:
    print(f"Processing batch of {len(data)} items")
    return ", ".join(data)
```

### Port Groups (Dot Notation)

Use dots in output port names to create collapsible groups in the UI:

```python
def extract_features(item: str) -> {"features.color": str, "features.shape": str, "features.size": str}:
    return {
        "features.color": "red",
        "features.shape": "circle",
        "features.size": "large",
    }
```

### `_node_config` Attribute

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

### Import Path Formats

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

### Factory Args in Configuration

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

---

## 5. Graph Topology

### Edges

Edges connect output ports to input ports. Use the shorthand string format `"NodeName.port_name"`:

```json
{
  "edges": [
    {"source_str": "double.out", "target_str": "add.a"},
    {"source_str": "add.out", "target_str": "format.value"}
  ]
}
```

TOML equivalent:

```toml
[[graph.edges]]
source_str = "double.out"
target_str = "add.a"
```

### Subgraphs

Subgraphs encapsulate a group of nodes and edges behind exposed ports. They are flattened at resolution time — internally, all nodes live in the same namespace (prefixed with the subgraph name).

#### Inline Subgraph

```json
{
  "type": "subgraph",
  "name": "preprocess",
  "nodes": [
    {
      "name": "normalize",
      "factory": "netrun.node_factories.from_function",
      "factory_args": {"func": "nodes.normalize"}
    },
    {
      "name": "validate",
      "factory": "netrun.node_factories.from_function",
      "factory_args": {"func": "nodes.validate"}
    }
  ],
  "edges": [
    {"source_str": "normalize.out", "target_str": "validate.data"}
  ],
  "exposed_in_ports": {
    "in": {"internal_node": "normalize", "internal_port": "data"}
  },
  "exposed_out_ports": {
    "out": {"internal_node": "validate", "internal_port": "out"}
  }
}
```

External edges connect to exposed ports: `"source_str": "source.out", "target_str": "preprocess.in"`.

#### File-Referenced Subgraph

```json
{
  "type": "subgraph",
  "name": "shared",
  "path": "./shared_pipeline.netrun.json",
  "exposed_in_ports": {
    "in": {"internal_node": "validate", "internal_port": "data"}
  },
  "exposed_out_ports": {
    "out": {"internal_node": "enrich", "internal_port": "out"}
  }
}
```

The `path` is relative to `project_root`. The referenced file contains `nodes` and `edges` arrays.

#### Factory-Generated Subgraph

A factory module can return a `SubgraphConfig` instead of a `NodeConfig`:

```json
{
  "name": "factory_pipeline",
  "factory": "./pipeline_factory.py",
  "factory_args": {"num_stages": 3}
}
```

```python
# pipeline_factory.py
from netrun.core import NodeConfig, EdgeConfig
from netrun.net.config import SubgraphConfig, ExposedPortConfig

def get_node_config(_net_config=None, *, num_stages: int = 2):
    nodes = [
        NodeConfig(
            name=f"stage_{i}",
            factory="netrun.node_factories.from_function",
            factory_args={"func": "nodes.process_stage"},
        )
        for i in range(num_stages)
    ]
    edges = [
        EdgeConfig(source_str=f"stage_{i}.out", target_str=f"stage_{i+1}.data")
        for i in range(num_stages - 1)
    ]
    return SubgraphConfig(
        name="pipeline",
        nodes=nodes,
        edges=edges,
        exposed_in_ports={"in": ExposedPortConfig(internal_node="stage_0", internal_port="data")},
        exposed_out_ports={"out": ExposedPortConfig(internal_node=f"stage_{num_stages-1}", internal_port="out")},
    )

# No get_node_funcs needed for subgraph factories
```

---

## 6. Ports & Flow Control

### Port Slot Specs

Each port has a **slot spec** that limits how many packets it can hold at once.

| Type | JSON | Description |
|------|------|-------------|
| Infinite (default) | `{"type": "infinite"}` | No limit on queued packets |
| Finite | `{"type": "finite", "capacity": 5}` | At most N packets |

```json
{
  "name": "batch_processor",
  "factory": "netrun.node_factories.from_function",
  "factory_args": {"func": "nodes.batch_processor"},
  "in_ports": {
    "data": {
      "slots_spec": {"type": "finite", "capacity": 5}
    }
  }
}
```

### Salvo Conditions

Salvo conditions define **when** a node fires and **which** packets participate.

By default, the function factory generates:
- **Input salvo**: fires when all input ports have at least 1 packet (consumes 1 per non-list port, all for list ports).
- **Output salvo**: fires once, sending all output packets.

#### Custom Input Salvo Conditions

Override for batching, accumulation, or conditional triggering:

```json
{
  "in_salvo_conditions": {
    "trigger": {
      "max_salvos": {"type": "finite", "max": 1},
      "ports": {"data": {"type": "count", "count": 3}},
      "term": {
        "type": "port",
        "port_name": "data",
        "state": {"type": "equals_or_greater_than", "value": 3}
      }
    }
  }
}
```

This fires when port `data` has 3 or more packets, consuming exactly 3.

#### Salvo Condition Terms

Terms are boolean expressions over port states:

| Term Type | Description |
|-----------|-------------|
| `true` / `false` | Always true / always false |
| `port` | Check a port's state (see below) |
| `and` | All sub-terms must be true |
| `or` | Any sub-term must be true |
| `not` | Negate a sub-term |

#### Port State Predicates

| State | Description |
|-------|-------------|
| `empty` | Port has no packets |
| `non_empty` | Port has at least one packet |
| `full` | Port is at capacity (finite slots only) |
| `non_full` | Port is not at capacity |
| `equals` | Packet count equals value |
| `less_than` / `greater_than` | Comparison with value |
| `equals_or_less_than` / `equals_or_greater_than` | Comparison with value |

#### Packet Count Modes

| Mode | JSON | Description |
|------|------|-------------|
| All | `{"type": "all"}` | Take all packets from the port |
| Count N | `{"type": "count", "count": 3}` | Take at most N packets |

### Type Checking

When `type_checking_enabled` is `true` (the default), netrun validates that packet values match port type annotations. A mismatch raises `PacketTypeMismatch`.

Supported types: Python types (`int`, `str`, `float`, etc.), generic aliases (`list[int]`), and custom classes. Validation uses beartype.

---

## 7. The Net Class

`Net` is the main runtime class. It manages the simulation, pools, packet storage, and epoch execution.

### Creating a Net

```python
from netrun.core import Net, NetConfig

# From file
net = Net(NetConfig.from_file("main.netrun.json"))

# Class method shorthand
net = Net.from_file("main.netrun.json")
```

### Lifecycle

```python
# Context manager (recommended)
async with Net(config) as net:
    # net is started, will be stopped on exit
    ...

# Manual lifecycle
net = Net(config)
await net.start()
# ... use net ...
await net.stop()

# Synchronous wrappers
net.start_sync()
net.stop_sync()
```

### Injecting Data

```python
# Create packet and inject in one step
net.inject_data("node_name", "port_name", [value1, value2])

# Or manually
packet_id = net.create_external_packet(value)
net.inject_packet(packet_id, "node_name", "port_name")

# Via NodeInfo helper
net.nodes["my_node"].inject_packet("port_name", value)
net.nodes["my_node"].inject_packets("port_name", [val1, val2])
net.nodes["my_node"].inject({"port_a": val1, "port_b": val2})
# Inject multiple values per port:
net.nodes["my_node"].inject({"port_a": [v1, v2]}, plural=True)
```

### Execution Loop

The standard execution pattern:

```python
async with Net(config) as net:
    net.inject_data("source", "in", [data])

    while True:
        await net.run_until_blocked()
        startable = net.get_startable_epochs()
        if not startable:
            break
        for epoch_id in startable:
            await net.execute_epoch(epoch_id)
```

For convenience, use `execute_startable_epochs()` to execute all at once:

```python
while True:
    await net.run_until_blocked()
    executed = await net.execute_startable_epochs()
    if not executed:
        break
```

### Background Execution

For fire-and-forget scenarios:

```python
async with Net(config) as net:
    await net.start_background()
    net.inject_data("source", "in", [data])
    await net.wait_until_done()
    results = net.flush_output_queue("results")
```

Background mode runs the execution loop automatically. Use `pause()` / `resume()` to control it.

### Output Queues

Output queues capture packets from unconnected output ports:

```json
{
  "output_queues": {
    "results": {"ports": [["format", "out"]]},
    "debug":   {"ports": [["node_a", "debug"], ["node_b", "debug"]]}
  }
}
```

Retrieve results:

```python
# Blocking wait
value = await net.get_output("results", timeout=5.0)

# Non-blocking
value = net.try_get_output("results")  # Returns None if empty

# Drain entire queue
values = net.flush_output_queue("results")

# Drain all queues
all_results = net.flush_all_output_queues()  # {"results": [...], "debug": [...]}

# With metadata (returns ConsumedOutputPacket objects)
packets = net.flush_output_queue("results", include_metadata=True)
for pkt in packets:
    print(pkt.value, pkt.from_node, pkt.from_port, pkt.timestamp, pkt.epoch_id)

# Query
net.has_output("results")    # bool
net.output_count("results")  # int
net.list_output_queues()     # ["results", "debug"]
```

---

## 8. Node Execution Context

The `ctx` parameter provides access to the execution context inside a node function.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `ctx.epoch_id` | `str` | Unique epoch identifier |
| `ctx.node_name` | `str` | Name of the current node |
| `ctx.retry_count` | `int` | Current retry attempt (0 on first try) |
| `ctx.retry_timestamps` | `list[datetime]` | Timestamps of previous retry attempts |
| `ctx.retry_exceptions` | `list[Exception]` | Exceptions from previous retries |
| `ctx.vars` | `dict[str, Any]` | Resolved node variables (see [Node Variables](#11-node-variables)) |

### Packet Operations

For advanced use (when `manual_output=True` or for custom packet management):

```python
def my_node(data: str, ctx):
    # Create a new packet
    packet_id = ctx.create_packet("some value")

    # Create packet with a lazy value (function resolved on access)
    packet_id = ctx.create_packet_from_value_func("mymodule.expensive_func", args=(1,), kwargs={})

    # Consume a packet
    value = ctx.consume_packet(packet_id)

    # Load packet into an output port
    ctx.load_output_port("out", packet_id)

    # Send output salvo
    ctx.send_output_salvo("send")
```

### Print Capture

```python
def my_node(data: str, print):
    print(f"Processing: {data}")   # Logged with timestamp
    print("Step 1 done")           # Appears in epoch logs
```

### Cancel Epoch

```python
def my_node(data: str, ctx) -> str:
    if data == "invalid":
        ctx.cancel_epoch()  # Raises EpochCancelled, discards all in-flight packets
    return data.upper()
```

---

## 9. Execution Configuration

Each node can have an `execution_config` section controlling how it runs.

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

### Key Fields

| Field | Default | Description |
|-------|---------|-------------|
| `pools` | `["main"]` | Which pool(s) can execute this node |
| `retries` | `0` | Number of retry attempts on failure |
| `retry_wait` | `0.0` | Seconds to wait between retries |
| `timeout` | `null` | Max seconds for execution (requires thread/process pool) |
| `max_epochs` | `null` | Total epoch limit for this node |
| `max_parallel_epochs` | `null` | Max concurrent epochs |
| `rate_limit_per_second` | `null` | Max epochs per second |
| `capture_prints` | `true` | Capture print output with timestamps |
| `print_flush_interval` | `0.1` | How often to flush print buffer (seconds) |
| `print_buffer_max_size` | `null` | Max print buffer entries |
| `print_echo_stdout` | `false` | Also print to real stdout |
| `type_checking_enabled` | `null` | Override net-level type checking |
| `propagate_exceptions` | `null` | Override net-level exception propagation |
| `print_exceptions` | `null` | Override net-level exception printing |
| `defer_startup` | `false` | Defer `start_node_func` until first epoch |
| `pool_allocation_method` | `null` | Override allocation method |
| `on_node_failure` | `null` | Callback function on failure (import path or callable) |
| `cache` | `null` | Per-node cache overrides (see [Caching](#12-caching--memoization)) |

### Net-Level Defaults

Some settings can be set at the net config level as defaults:

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

---

## 10. Worker Pools

Pools determine **where** node functions execute. netrun supports four pool types.

### Pool Types

| Type | Description | Use Case |
|------|-------------|----------|
| `main` | Single async worker in the main event loop | I/O-bound work, default |
| `thread` | Multiple worker threads in the same process | Blocking I/O, timeouts |
| `multiprocess` | Separate subprocesses with worker threads | CPU-bound work |
| `remote` | Workers on remote machines via WebSocket | Distributed execution |

### Configuration

```json
{
  "pools": {
    "main": {
      "spec": {"type": "main"}
    },
    "threads": {
      "spec": {"type": "thread", "num_workers": 4}
    },
    "processes": {
      "spec": {
        "type": "multiprocess",
        "num_processes": 2,
        "threads_per_process": 2
      }
    },
    "remote": {
      "spec": {
        "type": "remote",
        "url": "ws://192.168.1.100:8765",
        "worker_name": "execution_manager",
        "num_processes": 1,
        "threads_per_process": 1
      }
    }
  }
}
```

If no pools are defined, a default `"main"` pool is created automatically.

### Assigning Nodes to Pools

```json
{
  "execution_config": {
    "pools": ["threads"]
  }
}
```

A node can list multiple pools. The allocation method determines which one is used:

```json
{
  "execution_config": {
    "pools": ["processes", "threads", "main"],
    "pool_allocation_method": "least-busy"
  }
}
```

### Allocation Methods

| Method | Description |
|--------|-------------|
| `round-robin` | Cycles through pools in order (default) |
| `random` | Randomly selects a pool |
| `least-busy` | Picks the pool with the fewest running tasks |

### Remote Pool Server

Start a remote pool server from your net config:

```python
server_ctx = Net.serve_pool(config, host="0.0.0.0", port=8765)
async with server_ctx:
    # Server is running, clients can connect
    await asyncio.Future()  # Run forever
```

---

## 11. Node Variables

Node variables provide key-value configuration accessible inside node functions via `ctx.vars`.

### Global Variables

Defined at the net config level, available to all nodes:

```json
{
  "node_vars": {
    "label":   {"value": "primes", "type": "str"},
    "verbose": {"value": "false",  "type": "bool"}
  }
}
```

### Per-Node Overrides

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

### Variable Types

| Type | Example Value | Resolved Python Type |
|------|---------------|---------------------|
| `str` | `"hello"` | `str` |
| `int` | `"42"` | `int` |
| `float` | `"3.14"` | `float` |
| `bool` | `"true"` / `"false"` | `bool` |
| `json` | `'{"key": "value"}'` | parsed via `json.loads` |

### Accessing in Node Functions

```python
def find_primes(n: int, ctx, print) -> list:
    label = ctx.vars.get("label", "default")
    verbose = ctx.vars.get("verbose", False)
    if verbose:
        print(f"[{label}] Finding primes up to {n}")
    return [x for x in range(2, n) if all(x % i for i in range(2, x))]
```

---

## 12. Caching & Memoization

netrun can cache epoch inputs and outputs so repeated inputs skip execution entirely.

### Enabling Caching

At the net config level:

```json
{
  "cache": {
    "enabled": true,
    "include_all_nodes": true
  }
}
```

Or selectively cache specific nodes:

```json
{
  "cache": {
    "enabled": true,
    "include_nodes": ["expensive_node", "transform_*"]
  }
}
```

### Cache Modes

| Mode | Description |
|------|-------------|
| `both` (default) | On hit: skip execution, replay cached output. On miss: execute and store. |
| `output` | Always execute. Record outputs for inspection. |
| `input` | Always execute. Record inputs for replay testing. |

### Per-Node Overrides

```json
{
  "execution_config": {
    "cache": {
      "enabled": true,
      "cache_what": "output",
      "version": 2
    }
  }
}
```

### Version-Based Invalidation

Change the version number to invalidate all cached entries:

```json
{
  "cache": {
    "enabled": true,
    "version": 2
  }
}
```

### Persistent Storage

By default, cache is stored in a temporary directory and lost between runs. Set `storage_path` to persist:

```json
{
  "cache": {
    "enabled": true,
    "storage_path": "./.cache/netrun"
  }
}
```

### Cache API

```python
# Check if caching is enabled for a node
net.nodes["my_node"].is_cache_enabled

# Get all cached entries for a node
entries = net.get_cached_entries("my_node")

# Get cached input/output salvos
inputs = net.get_cached_input_salvos("my_node")
outputs = net.get_cached_output_salvos("my_node")

# Look up cached output for specific input
cached = net.get_cached_output_for_input("my_node", {"port_a": [value1]})

# Cache statistics
stats = net.cache_stats()  # {"my_node": {"entry_count": 5, ...}}

# Clearing
net.clear_cache()                                           # Clear all
net.clear_node_cache("my_node")                             # Clear one node
net.clear_cache_by_version("my_node", net_version=1)        # Clear old version
net.clear_cached_output_for_input("my_node", {"port_a": [val]})  # Clear specific
net.clear_cached_inputs("my_node")                          # Clear input-only entries
```

### Detecting Cache Hits

```python
for epoch_id, epoch in net.epochs.items():
    if epoch.was_cache_hit:
        print(f"{epoch.node_name}: served from cache")
```

Also available via NodeInfo:

```python
node = net.nodes["my_node"]
print(node.cache_stats)
print(node.cached_entries)
```

---

## 13. Error Handling

### Retries

Configure automatic retries with optional wait time:

```json
{
  "execution_config": {
    "retries": 3,
    "retry_wait": 0.5
  }
}
```

Inside the node, `ctx.retry_count` tracks the current attempt:

```python
def flaky_node(data: str, ctx, print) -> str:
    if ctx.retry_count < 2:
        raise ValueError(f"Transient error (attempt {ctx.retry_count + 1})")
    print(f"Attempt {ctx.retry_count + 1}: success!")
    return data.upper()
```

### On-Failure Callback

Called after each failed attempt (before retry or final failure):

```json
{
  "execution_config": {
    "on_node_failure": "nodes.on_failure"
  }
}
```

```python
def on_failure(ctx):
    """ctx is a NodeFailureContext with epoch_id, node_name, exception, retry_count, etc."""
    ctx.print(f"[on_failure] '{ctx.node_name}' failed: {ctx.exception}")
```

### Epoch Cancellation

A node can cancel its own epoch, discarding all in-flight packets:

```python
def my_node(data: str, ctx) -> str:
    if data == "invalid":
        ctx.cancel_epoch()  # Raises EpochCancelled
    return data
```

### Exception Propagation

By default, exceptions propagate and stop the network. Set `propagate_exceptions: false` to queue them instead:

```json
{
  "execution_config": {
    "propagate_exceptions": false
  }
}
```

Queued exceptions can be inspected later:

```python
# Check exception queue
for exc in net.exception_queue:
    print(exc)

# Or raise all at once
net.propagate_exceptions()  # Raises ExceptionGroup
```

### Dead Letter Queue

Failed epochs (after exhausting retries) are sent to the dead letter queue:

```python
for entry in net.dead_letter_queue:
    print(entry)

# Drain and return
entries = net.clear_dead_letter_queue()
```

### Timeouts

Timeouts require a thread or process pool (the main pool runs in the event loop and cannot interrupt):

```json
{
  "execution_config": {
    "pools": ["threads"],
    "timeout": 5.0
  }
}
```

### Type Checking

Enable at the net or node level to validate packet types against port annotations:

```json
{
  "type_checking_enabled": true
}
```

A type mismatch raises `PacketTypeMismatch`.

---

## 14. Inspecting Results

### NodeInfo

Access node information and state through `net.nodes`:

```python
node = net.nodes["my_node"]

# Configuration
node.name                  # "my_node"
node.in_port_names         # ["data", "config"]
node.out_port_names        # ["out"]
node.execution_config      # NodeExecutionConfig
node.pools                 # ["main"]

# State
node.epochs                # List of EpochRecord objects
node.epoch_count           # Total epochs executed
node.running_epochs        # Currently executing
node.startable_epochs      # Ready to execute
node.is_busy               # Has running epochs

# Edges
node.incoming_edges        # List of EdgeInfo
node.outgoing_edges        # List of EdgeInfo

# Packets
node.packets_at_input_port("data")       # Packets at a port
node.packets_at_all_input_ports()        # All input ports
```

### EdgeInfo

```python
for edge in net.edges:
    print(f"{edge.source_node}.{edge.source_port} -> {edge.target_node}.{edge.target_port}")
    print(f"  Packets in transit: {edge.packet_count}")
```

### Epoch Logs

```python
# Per-epoch
net.print_epoch_logs(epoch_id)

# Per-node
net.print_node_logs("my_node")
net.print_node_logs("my_node", chronological=True)

# All logs
net.print_all_logs()
net.print_all_logs(chronological=True)

# Programmatic access
logs = net.get_epoch_log(epoch_id)        # [(timestamp, message), ...]
logs = net.get_node_logs("my_node")       # [(timestamp, message), ...]
all_logs = net.get_all_logs()             # {node: {epoch: [(ts, msg), ...]}}
all_logs = net.get_all_logs_chronological()  # [(ts, msg), ...] sorted
```

### Epoch Records

```python
for epoch_id, epoch in net.epochs.items():
    print(f"Node: {epoch.node_name}")
    print(f"State: {epoch.state}")
    print(f"Created: {epoch.created_at}")
    print(f"Started: {epoch.started_at}")
    print(f"Ended: {epoch.ended_at}")
    print(f"Cache hit: {epoch.was_cache_hit}")
    print(f"Cancelled: {epoch.was_cancelled}")
    print(f"Pool: {epoch.pool_id}, Worker: {epoch.worker_id}")
```

---

## 15. Targeted Execution

`run_to_targets` executes only the upstream portion of a graph needed to feed a specific target node, **without executing the target itself**. Useful for testing and debugging.

```python
async with Net(config) as net:
    net.inject_data("source", "in", [data])

    # Run everything upstream of "transform" but don't execute it
    salvos = await net.run_to_targets("transform")

    for salvo in salvos:
        print(f"Target: {salvo.node_name}")
        for port_name, values in salvo.packets.items():
            for value in values:
                print(f"  Port '{port_name}': {value}")
```

Targets can be node names or (node_name, salvo_condition) tuples. Multiple targets are supported:

```python
salvos = await net.run_to_targets(["transform", "validate"])
```

Use cases:
- Inspect what data a node would receive without executing it.
- Test a node function in isolation with real upstream data.
- Debug data transformation issues at specific pipeline stages.

---

## 16. Actions & Recipes

### Actions

Actions are shell commands with template variable substitution. They are defined in the `extra.ui.actions` section and can be run via the CLI or the UI.

#### Project-Level Actions

```json
{
  "graph": {
    "extra": {
      "ui": {
        "actions": [
          {
            "id": "action-show-info",
            "label": "Show Node Info",
            "command": "echo \"Node: $NODE_NAME\" && echo \"App: $APP_NAME\""
          }
        ],
        "env": {
          "APP_NAME": "my-app"
        }
      }
    }
  }
}
```

#### Node-Level Actions

```json
{
  "name": "my_node",
  "extra": {
    "ui": {
      "actions": [
        {
          "id": "action-test",
          "label": "Test This Node",
          "command": "echo \"Testing $NODE_NAME ($ROLE)\""
        }
      ],
      "env": {
        "ROLE": "processor"
      }
    }
  }
}
```

#### Template Variables

| Variable | Description |
|----------|-------------|
| `$NODE_NAME` | Current node name |
| `$NET_FILE_PATH` | Full path to config file |
| `$NET_FILE_DIR` | Directory containing config file |
| `$PROJECT_ROOT` | Project root directory |
| `$DEFAULT_CMD` | Default command from `ui.defaultCmd` |
| `$NODE_CONFIG` | Node config as JSON string |
| Custom variables | From `ui.env` (project and node level) and `node_vars` |

#### Running Actions via CLI

```bash
netrun actions list
netrun actions list --node my_node
netrun actions run action-show-info my_node
netrun actions run action-show-info --global
```

### Recipes

Recipes are Python scripts that transform the net configuration based on user inputs.

#### Defining Recipes

```json
{
  "recipes": {
    "add_node": {
      "path": "./recipes/add_node.py",
      "description": "Add a new node to the graph"
    }
  }
}
```

#### Recipe Python File

```python
# recipes/add_node.py

def get_prompts(config):
    """Return prompts for user input (optional)."""
    return [
        {"name": "node_name", "label": "Node name", "type": "text"},
        {"name": "pool", "label": "Pool", "type": "select", "options": ["main", "threads"], "default": "main"},
    ]

def run(config, inputs):
    """Transform the config based on user inputs. Returns modified config dict."""
    new_node = {
        "name": inputs["node_name"],
        "factory": "netrun.node_factories.from_function",
        "factory_args": {"func": f"nodes.{inputs['node_name']}"},
        "execution_config": {"pools": [inputs["pool"]]}
    }
    config["graph"]["nodes"].append(new_node)
    return config
```

#### Prompt Types

| Type | Description |
|------|-------------|
| `text` | Free-form text input |
| `number` | Numeric input |
| `select` | Single selection from `options` list |
| `checkbox` | Multi-selection from `options` list |

#### Running Recipes via CLI

```bash
netrun recipes list
netrun recipes run add_node -i '{"node_name": "transform", "pool": "threads"}' -o main.netrun.json
```

---

## 17. CLI Reference

The `netrun` CLI provides commands for inspecting and managing netrun projects.

### Core Commands

| Command | Description |
|---------|-------------|
| `netrun validate [-c CONFIG]` | Validate a netrun config file |
| `netrun info [-c CONFIG]` | Summary statistics (node count, pools, factories, etc.) |
| `netrun structure [-c CONFIG]` | Output graph topology as JSON |
| `netrun nodes [-c CONFIG]` | List all nodes with port names |
| `netrun node NAME [-c CONFIG]` | Detailed info about a specific node |
| `netrun convert FILE [-o OUTPUT]` | Convert between JSON and TOML formats |
| `netrun factory-info FACTORY_PATH` | Inspect a factory module's parameters |

### Actions Commands

| Command | Description |
|---------|-------------|
| `netrun actions list [-c CONFIG] [-n NODE]` | List available actions |
| `netrun actions run ACTION_ID [NODE] [-c CONFIG] [-g] [-t TIMEOUT]` | Execute an action |

### Recipes Commands

| Command | Description |
|---------|-------------|
| `netrun recipes list [-c CONFIG]` | List available recipes |
| `netrun recipes run NAME [-c CONFIG] [-i INPUTS_JSON] [-o OUTPUT]` | Execute a recipe |

### Common Options

| Option | Description |
|--------|-------------|
| `-c, --config PATH` | Config file path (auto-detects `.netrun.json`/`.toml` in cwd if omitted) |
| `--pretty / --compact` | JSON output formatting (default: pretty) |

---

## 18. netrun-ui

**netrun-ui** is a visual editor for creating and editing netrun flow configurations.

### Installation

```bash
pip install netrun-ui
# or
uv add netrun-ui
```

### Starting the UI

```bash
netrun-ui                              # Native window (background)
netrun-ui main.netrun.json             # Open a specific file
netrun-ui --fg                         # Foreground (blocks until closed)
netrun-ui --server                     # Browser mode (opens http://localhost:PORT)
netrun-ui --dev                        # Development mode with hot reload
```

### Options

| Option | Description |
|--------|-------------|
| `FILE` | Config file to open (`.netrun.json` or `.netrun.toml`) |
| `-s, --server` | Run in server/browser mode instead of native window |
| `--fg, --foreground` | Block until the window closes (default: background) |
| `-d, --dev` | Development mode using Vite dev server |
| `-p, --port PORT` | Backend port (default: auto-select 8000-8099) |
| `--frontend-port PORT` | Frontend dev server port (default: 5173, `--dev` only) |
| `-C, --working-dir PATH` | Working directory for file explorer |
| `--width WIDTH` | Window width in pixels (default: 1400) |
| `--height HEIGHT` | Window height in pixels (default: 900) |

### Modes

| Mode | Command | Description |
|------|---------|-------------|
| Native (background) | `netrun-ui` | Opens native window, returns control to terminal |
| Native (foreground) | `netrun-ui --fg` | Opens native window, blocks until closed |
| Server | `netrun-ui --server` | Browser-based, production build |
| Dev | `netrun-ui --dev --fg` | Native window with Vite hot reload |
| Dev server | `netrun-ui --dev --server` | Browser with Vite hot reload |

---

## 19. Configuration Reference Tables

### NetConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project_root` | `str` | config file dir | Base path for relative imports |
| `graph` | `GraphConfig` | *required* | Graph definition |
| `pools` | `dict[str, PoolConfig]` | auto `main` | Worker pool definitions |
| `output_queues` | `dict[str, OutputQueueConfig]` | `null` | Output queue mappings |
| `node_vars` | `dict[str, NodeVariable]` | `null` | Global node variables |
| `cache` | `CacheConfig` | `null` | Caching configuration |
| `type_checking_enabled` | `bool` | `true` | Validate packet types against port annotations |
| `propagate_exceptions` | `bool` | `true` | Raise exceptions vs. queue them |
| `print_exceptions` | `bool` | `false` | Print exceptions to stderr |
| `dead_letter_queue` | `bool` | `true` | Enable dead letter queue for failed epochs |
| `dead_letter_path` | `str` | `null` | File path to persist dead letter queue |
| `dead_letter_callback` | `str` | `null` | Callback function for dead letter entries |
| `default_pool_allocation_method` | `str` | `round-robin` | Default pool selection strategy |
| `error_on_undeclared_output` | `bool` | `false` | Error on output to undeclared queue port |
| `recipes` | `dict[str, RecipeConfig]` | `null` | Recipe definitions |
| `extra` | `dict` | `{}` | Arbitrary metadata (UI settings, actions, etc.) |

### NodeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `""` | Node name (required in practice) |
| `factory` | `str` | `null` | Factory module path |
| `factory_args` | `dict` | `{}` | Arguments passed to the factory |
| `in_ports` | `dict[str, PortConfig]` | `{}` | Input port definitions (usually auto-generated by factory) |
| `out_ports` | `dict[str, PortConfig]` | `{}` | Output port definitions (usually auto-generated by factory) |
| `in_salvo_conditions` | `dict[str, SalvoConditionConfig]` | auto | Input salvo conditions |
| `out_salvo_conditions` | `dict[str, SalvoConditionConfig]` | auto | Output salvo conditions |
| `execution_config` | `NodeExecutionConfig` | `null` | Execution settings |
| `extra` | `dict` | `{}` | Arbitrary metadata |

### NodeExecutionConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pools` | `list[str]` | `["main"]` | Pool(s) that can execute this node |
| `retries` | `int` | `0` | Retry attempts on failure |
| `retry_wait` | `float` | `0.0` | Seconds between retries |
| `timeout` | `float` | `null` | Max execution time (seconds) |
| `max_epochs` | `int` | `null` | Total epoch limit |
| `max_parallel_epochs` | `int` | `null` | Max concurrent epochs |
| `rate_limit_per_second` | `float` | `null` | Epoch rate limit |
| `capture_prints` | `bool` | `true` | Capture print output |
| `print_flush_interval` | `float` | `0.1` | Print buffer flush interval |
| `print_buffer_max_size` | `int` | `null` | Max print buffer entries |
| `print_echo_stdout` | `bool` | `false` | Echo prints to real stdout |
| `type_checking_enabled` | `bool` | `null` | Per-node type checking override |
| `propagate_exceptions` | `bool` | `null` | Per-node exception propagation override |
| `print_exceptions` | `bool` | `null` | Per-node exception printing override |
| `defer_startup` | `bool` | `false` | Defer `start_node_func` until first epoch |
| `pool_allocation_method` | `str` | `null` | Per-node pool allocation override |
| `node_vars` | `dict[str, NodeVariable]` | `null` | Per-node variable overrides |
| `on_node_failure` | `str` | `null` | Failure callback (import path) |
| `exec_node_func` | `str` | `null` | Custom execution function (import path) |
| `start_node_func` | `str` | `null` | Called on pool start (import path) |
| `stop_node_func` | `str` | `null` | Called on pool stop (import path) |
| `cache` | `NodeCacheConfig` | `null` | Per-node cache overrides |

### PortConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `slots_spec` | `PortSlotSpecConfig` | infinite | Slot capacity (`{"type": "infinite"}` or `{"type": "finite", "capacity": N}`) |
| `port_type` | `str \| type` | `null` | Type annotation for validation |

### CacheConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | *required* | Enable caching |
| `version` | `int` | `0` | Cache version (change to invalidate) |
| `storage_path` | `str` | `null` | Persistent storage directory (null = temp dir) |
| `include_all_nodes` | `bool` | `false` | Cache all nodes |
| `include_nodes` | `list[str]` | `null` | Glob patterns of nodes to cache |
| `exclude_nodes` | `list[str]` | `null` | Glob patterns to exclude |
| `cache_what` | `str` | `"both"` | `"both"`, `"output"`, or `"input"` |
| `hash_method` | `str` | `"xxh64"` | Hash algorithm |
| `pickling_method` | `str` | `"pickle"` | Serialization method |
| `sample_size` | `int` | `null` | Max cached entries per node (reservoir sampling) |

### NodeCacheConfig

Same fields as CacheConfig (except `include_all_nodes`, `include_nodes`, `exclude_nodes`), all optional. Non-null values override the net-level CacheConfig for that node.

### PoolConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `spec` | `PoolSpecConfig` | main | Pool type specification (see below) |
| `print_flush_interval` | `float` | `0.1` | Print buffer flush interval |
| `capture_prints` | `bool` | `true` | Capture worker print output |

### Pool Spec Types

**MainPoolConfig**: `{"type": "main"}`

**ThreadPoolConfig**: `{"type": "thread", "num_workers": 1}`

**MultiprocessPoolConfig**: `{"type": "multiprocess", "num_processes": 1, "threads_per_process": 1}`

**RemotePoolConfig**: `{"type": "remote", "url": "ws://...", "worker_name": "...", "num_processes": 1, "threads_per_process": 1}`

### NodeVariable

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `value` | `str` | *required* | Variable value (always stored as string) |
| `type` | `str` | `"str"` | Type for resolution: `"str"`, `"int"`, `"float"`, `"bool"`, `"json"` |

### OutputQueueConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ports` | `list[[str, str]]` | *required* | List of `[node_name, port_name]` tuples to capture |

---

## 20. Sample Project Index

The `sample_projects/` directory contains working examples of every major netrun feature. **When you need concrete implementation details beyond what this document provides, read the relevant sample project files directly.** Each project is self-contained with a config file, node functions, and an execution script.

### Project Directory

| # | Project | Path | Description |
|---|---------|------|-------------|
| 00 | Basic Net Project | `sample_projects/00_basic_net_project/` | Foundational example — start here |
| 01 | Thread and Process Pools | `sample_projects/01_thread_and_process_pools/` | Worker pool types and allocation |
| 02 | Remote Deployment | `sample_projects/02_remote_deployment/` | Cloud deployment with remote pools |
| 03 | Subgraphs | `sample_projects/03_subgraphs/` | Graph composition and encapsulation |
| 04 | Error Handling | `sample_projects/04_error_handling/` | Retries, timeouts, failure callbacks |
| 05 | Advanced Flow Control | `sample_projects/05_advanced_flow_control/` | Custom salvos, batching, rate limiting |
| 06 | Actions and Recipes | `sample_projects/06_actions_and_recipes/` | TOML config, actions, recipes, file-path imports |
| 07 | Run to Targets | `sample_projects/07_run_to_targets/` | Targeted execution for testing |
| 08 | Caching | `sample_projects/08_caching/` | Memoization and cache management |

### Concept-to-Project Index

The table below maps every netrun concept and API to the sample project(s) that demonstrate it. Use this to find working examples quickly.

#### Core Concepts

| Concept / API | Sample Projects |
|---------------|-----------------|
| Function factory (`netrun.node_factories.from_function`) | 00, 01, 03, 04, 05, 06, 07, 08 |
| `NetConfig.from_file()` | 00, 01, 04, 05, 07, 08 |
| JSON config format | 00, 01, 03, 04, 05, 07, 08 |
| TOML config format | 06 |
| `project_root` | 06 |
| Env var substitution (`$env`) | 06 |
| File-path imports (`./nodes.py::func`) | 06 |
| Dotted import paths (`nodes.my_func`) | 00, 01, 04, 05, 07, 08 |

#### Nodes & Ports

| Concept / API | Sample Projects |
|---------------|-----------------|
| Single output port (`-> type`) | 00, 01, 04, 07 |
| Multiple output ports (`-> {"a": T, "b": T}`) | 00, 03, 05 |
| Port groups / dot notation (`features.color`) | 03 |
| `list[T]` input ports (batch consumption) | 05 |
| Special parameter: `print` | 00, 01, 04, 05, 06, 07, 08 |
| Special parameter: `ctx` | 01, 04, 08 |
| `_node_config` attribute override | 00 |
| Custom salvo conditions | 05 |
| Finite port slots | 05 |
| Type checking (`type_checking_enabled`) | 04 |

#### Graph Topology

| Concept / API | Sample Projects |
|---------------|-----------------|
| Edges (`source_str` / `target_str`) | 00, 03, 06, 07, 08 |
| Inline subgraphs | 03 |
| File-referenced subgraphs (`path`) | 03 |
| Factory-generated subgraphs | 03 |
| Exposed ports (`exposed_in_ports` / `exposed_out_ports`) | 03 |

#### Execution & Lifecycle

| Concept / API | Sample Projects |
|---------------|-----------------|
| `Net` context manager (`async with`) | 00, 01, 04, 05, 07, 08 |
| Execution loop (`run_until_blocked` + `execute_epoch`) | 00, 01, 04, 05, 07, 08 |
| `inject_data()` | 00, 01, 04, 05, 07, 08 |
| Output queues (`flush_output_queue`) | 00, 01, 04, 05, 06, 08 |
| `run_to_targets()` | 07 |
| Background execution (`start_background`) | — (documented in API) |

#### Worker Pools

| Concept / API | Sample Projects |
|---------------|-----------------|
| Main pool | 00, 04, 05, 06 |
| Thread pool | 01, 04 |
| Multiprocess pool | 01 |
| Remote pool | 01, 02 |
| Pool assignment (`execution_config.pools`) | 01, 04, 05, 06 |
| Pool allocation method (`least-busy`, etc.) | 01 |
| Remote pool server (`Net.serve_pool`) | 02 |

#### Error Handling

| Concept / API | Sample Projects |
|---------------|-----------------|
| Retries (`retries`, `retry_wait`) | 04 |
| `ctx.retry_count` / `ctx.retry_exceptions` | 04 |
| `on_node_failure` callback | 04 |
| `ctx.cancel_epoch()` | 04 |
| `propagate_exceptions: false` | 04 |
| Exception queue (`net.exception_queue`) | 04 |
| Dead letter queue (`net.dead_letter_queue`) | 04 |
| Timeout (`timeout`) | 04 |
| `max_epochs` | 01, 04 |

#### Node Variables

| Concept / API | Sample Projects |
|---------------|-----------------|
| Global `node_vars` | 01, 06 |
| Per-node `node_vars` override | 01, 06 |
| `ctx.vars` access | 01, 06 |
| Variable types (`str`, `bool`, `int`, etc.) | 01, 06 |

#### Caching

| Concept / API | Sample Projects |
|---------------|-----------------|
| `CacheConfig` (net-level) | 08 |
| `include_all_nodes` | 08 |
| Cache modes (`both`, `output`, `input`) | 08 |
| Per-node cache overrides (`NodeCacheConfig`) | 08 |
| Version-based invalidation | 08 |
| Persistent storage (`storage_path`) | 08 |
| Cache API (`cache_stats`, `get_cached_*`, `clear_*`) | 08 |
| `epoch.was_cache_hit` | 08 |
| NodeInfo cache helpers | 08 |

#### Actions & Recipes

| Concept / API | Sample Projects |
|---------------|-----------------|
| Project-level actions (`extra.ui.actions`) | 06 |
| Node-level actions | 06 |
| Template variables (`$NODE_NAME`, `$PROJECT_ROOT`, etc.) | 06 |
| Custom env variables (`extra.ui.env`) | 06 |
| Recipes (`recipes` config + Python scripts) | 01, 06 |
| Recipe prompts (`get_prompts`) | 06 |

#### Inspection & Logging

| Concept / API | Sample Projects |
|---------------|-----------------|
| `net.get_node_logs()` / `net.print_all_logs()` | 00, 04, 07, 08 |
| `net.nodes["name"]` (NodeInfo) | 08 |
| Epoch records (`net.epochs`) | 08 |
| `print_echo_stdout` | 01 |
| Rate limiting (`rate_limit_per_second`) | 05 |
