---
name: netrun-cli
description: Use netrun CLI commands to inspect, validate, and modify pipeline graph configs (add/remove nodes and edges, check structure, fan-out with broadcast). Invoke when adding nodes to the pipeline, rewiring edges, or inspecting the graph topology.
---

# netrun CLI — Graph Management & Inspection

A practical skill for building and managing netrun flow-based networks via the CLI. This covers both the read-only inspection commands and the write commands for modifying graph configs.

## How netrun Works (30-second version)

netrun is a flow-based runtime. You define a **graph** of **nodes** connected by **edges**. Data flows as **packets**:

1. **Nodes** are Python functions. Parameters become **input ports**, return type becomes **output port(s)**.
2. **Edges** connect one node's output port to another's input port: `A.out -> B.in`.
3. When enough packets arrive at a node's input ports (satisfying a **salvo condition**), the node fires an **epoch** (one execution).
4. Output packets travel along edges to downstream nodes. The cycle repeats.
5. Unconnected output ports can feed into **output queues** for collecting results.

**Config files** (`.netrun.json` or `.netrun.toml`) define nodes, edges, pools, and metadata. Node logic lives in separate Python files, referenced by import path.

### Minimal config structure

```json
{
  "output_queues": {
    "results": {"ports": [["node_name", "out"]]}
  },
  "graph": {
    "nodes": [
      {
        "name": "double",
        "factory": "netrun.node_factories.from_function",
        "factory_args": {"func": "nodes.double"},
        "extra": {"ui": {"position": {"x": 300, "y": 100}}}
      }
    ],
    "edges": [
      {"source_str": "A.out", "target_str": "B.in"}
    ]
  }
}
```

### Key rules

- **No fan-out**: An output port can connect to only one edge. To send to multiple targets, use `netrun.node_factories.broadcast`.
- **Factories**: Most nodes use `netrun.node_factories.from_function` which auto-generates ports from the function signature.
- **Special function params**: `ctx` (NodeExecutionContext) and `print` (captured logger) are not ports.
- **Multi-output**: `def f(x: int) -> {"a": str, "b": int}` creates two output ports.

---

## CLI Reference

All commands auto-discover config files in the current directory (files ending `.netrun.json`/`.netrun.toml`). Use `-c PATH` to specify explicitly.

### Inspection Commands

```bash
# Validate config (pydantic + graph structure + Rust sim)
netrun validate [-c CONFIG]

# Summary stats (node count, edges, pools, factories)
netrun info [-c CONFIG]

# Graph topology as JSON (nodes with ports, edges)
netrun structure [-c CONFIG]

# List all nodes with port names
netrun nodes [-c CONFIG]

# Detailed info about a specific node
netrun node NODE_NAME [-c CONFIG]

# Inspect a factory module's parameters
netrun factory-info netrun.node_factories.from_function

# Convert between JSON and TOML
netrun convert FILE [-o OUTPUT]
```

### Graph Mutation Commands

All mutation commands output JSON to stdout and print warnings/errors to stderr. They auto-validate after writing (disable with `--no-validate`).

#### add-node

```bash
# Add a factory-based node
netrun add-node my_node -f netrun.node_factories.from_function --factory-arg func=nodes.my_func

# Add a node with explicit ports (no factory)
netrun add-node my_node --in-ports x,y --out-ports result

# Specify UI position (default: auto-positioned right of rightmost node)
netrun add-node my_node -f netrun.node_factories.from_function --factory-arg func=nodes.f --position 500,300

# Pipe a full node dict from stdin
echo '{"factory": "netrun.node_factories.from_function", "factory_args": {"func": "nodes.f"}}' | netrun add-node my_node --json
```

#### remove-node

```bash
# Removes the node AND all edges that reference it
netrun remove-node old_node
```

Output: `{"removed": "old_node", "edges_removed": 2}`

#### edit-node

```bash
# Rename (updates all edge and output_queue references)
netrun edit-node old_name --rename new_name

# Add/remove ports
netrun edit-node my_node --add-in-port extra_input --remove-out-port unused

# Deep-merge arbitrary JSON into the node dict
netrun edit-node my_node --merge '{"execution_config": {"pools": ["gpu_pool"], "retries": 3}}'

# Merge from stdin
echo '{"extra": {"description": "Processes data"}}' | netrun edit-node my_node --merge-stdin
```

#### add-edge

```bash
# Connect two ports (format: Node.port)
netrun add-edge source_node.out target_node.in

# Add a dependency edge
netrun add-edge data_node.out processor.trigger --dependency
```

Warns on fan-out (same source port already has an edge) and suggests the broadcast factory.

#### remove-edge

```bash
netrun remove-edge source_node.out target_node.in
```

### Actions & Recipes

```bash
# List available actions
netrun actions list [-c CONFIG]

# Run an action on a node
netrun actions run ACTION_ID NODE_NAME [-c CONFIG]

# Run a global action
netrun actions run ACTION_ID --global [-c CONFIG]

# List recipes
netrun recipes list [-c CONFIG]

# Run a recipe
netrun recipes run RECIPE_NAME [-c CONFIG]
```

---

## Common Workflows

### Build a pipeline from scratch

```bash
# 1. Start with an empty config (create manually or from template)
# 2. Add nodes
netrun add-node fetch -f netrun.node_factories.from_function --factory-arg func=nodes.fetch
netrun add-node process -f netrun.node_factories.from_function --factory-arg func=nodes.process
netrun add-node save -f netrun.node_factories.from_function --factory-arg func=nodes.save

# 3. Connect them
netrun add-edge fetch.out process.data
netrun add-edge process.result save.data

# 4. Validate
netrun validate
```

### Add a node to an existing pipeline

```bash
# 1. Check what's there
netrun nodes
netrun structure

# 2. Add the new node
netrun add-node transform -f netrun.node_factories.from_function --factory-arg func=nodes.transform

# 3. Rewire: remove old edge, add two new ones
netrun remove-edge fetch.out process.data
netrun add-edge fetch.out transform.input
netrun add-edge transform.output process.data

# 4. Validate
netrun validate
```

### Fan-out: send one output to multiple nodes

You cannot connect the same output port to multiple edges directly. Use a broadcast node:

```bash
# Add a broadcast node
netrun add-node broadcast_data -f netrun.node_factories.broadcast \
  --factory-arg 'input_ports=["in"]' \
  --factory-arg 'output_ports=["out_a", "out_b"]'

# Wire it up
netrun add-edge source.out broadcast_data.in
netrun add-edge broadcast_data.out_a consumer_a.data
netrun add-edge broadcast_data.out_b consumer_b.data
```

### Rename a node

```bash
# This updates the node name AND all edge/output_queue references
netrun edit-node old_name --rename new_name
```

### Configure execution settings

```bash
# Add retries and pool assignment via --merge
netrun edit-node my_node --merge '{"execution_config": {"pools": ["gpu"], "retries": 3, "timeout": 30.0}}'
```

---

## Output Format

All commands output JSON to stdout. Warnings and errors go to stderr. This makes them composable:

```bash
# Get node names as a list
netrun nodes --compact | jq '.[].name'

# Check if config is valid (exit code 0 = valid)
netrun validate --compact > /dev/null 2>&1 && echo "OK" || echo "INVALID"

# Get edge count
netrun info --compact | jq '.edges'
```

---

## Node Function Conventions

For reference, here's how node functions map to config:

```python
# nodes.py

def double(x: int, print) -> int:
    """x -> input port 'x', return -> output port 'out'"""
    print(f"Doubling {x}")  # captured with timestamp
    return x * 2

def merge(a: str, b: str) -> str:
    """Two input ports: 'a' and 'b'. One output port: 'out'."""
    return f"{a} + {b}"

def split(data: str) -> {"left": str, "right": str}:
    """One input port: 'data'. Two output ports: 'left' and 'right'."""
    mid = len(data) // 2
    return {"left": data[:mid], "right": data[mid:]}

def batch_process(items: list[str], print) -> str:
    """list[T] annotation: consumes ALL packets at that port in one epoch."""
    print(f"Processing {len(items)} items")
    return ", ".join(items)
```

Config for a function node:
```json
{
  "name": "double",
  "factory": "netrun.node_factories.from_function",
  "factory_args": {"func": "nodes.double"}
}
```

The factory reads the function signature and auto-generates ports and salvo conditions.
