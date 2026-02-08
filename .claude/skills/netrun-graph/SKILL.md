---
name: netrun-graph
description: "Netrun graph topology: defining edges with source_str/target_str shorthand, subgraphs (inline, file-referenced, factory-generated), SubgraphConfig, ExposedPortConfig, and how subgraphs flatten into the node namespace."
---

# Graph Topology

## Edges

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

## Subgraphs

Subgraphs encapsulate a group of nodes and edges behind exposed ports. They are flattened at resolution time — internally, all nodes live in the same namespace (prefixed with the subgraph name).

### Inline Subgraph

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

### File-Referenced Subgraph

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

### Factory-Generated Subgraph

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

### How Subgraph Flattening Works

When a subgraph named `"preprocess"` contains nodes `"normalize"` and `"validate"`, after flattening they become `"preprocess/normalize"` and `"preprocess/validate"` in the top-level namespace. Internal edges are similarly prefixed. Exposed ports create edges from the external graph into the internal flattened nodes.

## Sample Projects

- **03** (`sample_projects/03_subgraphs/`): All three subgraph types (inline, file-referenced, factory-generated), exposed ports, port groups
- **06** (`sample_projects/06_actions_and_recipes/`): Edge definitions in TOML format
