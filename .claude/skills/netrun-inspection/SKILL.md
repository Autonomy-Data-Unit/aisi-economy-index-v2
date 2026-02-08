---
name: netrun-inspection
description: "Inspecting netrun network state: NodeInfo API (name, in_port_names, out_port_names, epochs, epoch_count, running_epochs, is_busy, incoming_edges, outgoing_edges, packets_at_input_port), EdgeInfo (source_node, target_node, packet_count), epoch logs (print_epoch_logs, print_node_logs, print_all_logs, get_epoch_log, get_node_logs, get_all_logs_chronological), and EpochRecord fields (state, created_at, started_at, ended_at, was_cache_hit, was_cancelled, pool_id, worker_id)."
---

# Inspecting Results

## NodeInfo

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

### NodeInfo Injection Helpers

```python
node = net.nodes["my_node"]

# Single value to a port
node.inject_packet("port_name", value)

# Multiple values to a port
node.inject_packets("port_name", [val1, val2])

# Dict of port -> value
node.inject({"port_a": val1, "port_b": val2})

# Dict of port -> list of values
node.inject({"port_a": [v1, v2]}, plural=True)
```

### NodeInfo Cache Helpers

```python
node.is_cache_enabled   # bool
node.cache_stats        # Stats dict
node.cached_entries     # All cached entries
```

## EdgeInfo

```python
for edge in net.edges:
    print(f"{edge.source_node}.{edge.source_port} -> {edge.target_node}.{edge.target_port}")
    print(f"  Packets in transit: {edge.packet_count}")
```

EdgeInfo fields:
- `source_node` — Source node name
- `source_port` — Source port name
- `target_node` — Target node name
- `target_port` — Target port name
- `packet_count` — Number of packets currently in transit on this edge

## Epoch Logs

### Printing Logs

```python
# Per-epoch
net.print_epoch_logs(epoch_id)

# Per-node
net.print_node_logs("my_node")
net.print_node_logs("my_node", chronological=True)

# All logs
net.print_all_logs()
net.print_all_logs(chronological=True)
```

### Programmatic Access

```python
# Single epoch's logs
logs = net.get_epoch_log(epoch_id)        # [(timestamp, message), ...]

# All logs for a node
logs = net.get_node_logs("my_node")       # [(timestamp, message), ...]

# All logs across all nodes
all_logs = net.get_all_logs()             # {node: {epoch: [(ts, msg), ...]}}
all_logs = net.get_all_logs_chronological()  # [(ts, msg), ...] sorted by time
```

## Epoch Records

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

EpochRecord fields:

| Field | Type | Description |
|-------|------|-------------|
| `node_name` | `str` | Which node this epoch belongs to |
| `state` | `str` | Current state (startable, running, finished, cancelled) |
| `created_at` | `datetime` | When the epoch was created |
| `started_at` | `datetime \| None` | When execution started |
| `ended_at` | `datetime \| None` | When execution ended |
| `was_cache_hit` | `bool` | Whether this epoch was served from cache |
| `was_cancelled` | `bool` | Whether this epoch was cancelled |
| `pool_id` | `str \| None` | Which pool executed this epoch |
| `worker_id` | `int \| None` | Which worker in the pool |

## Sample Projects

- **00** (`sample_projects/00_basic_net_project/`): `print_all_logs`, basic inspection
- **04** (`sample_projects/04_error_handling/`): Epoch logs with retries
- **07** (`sample_projects/07_run_to_targets/`): Inspecting upstream data
- **08** (`sample_projects/08_caching/`): NodeInfo, epoch records, cache inspection
