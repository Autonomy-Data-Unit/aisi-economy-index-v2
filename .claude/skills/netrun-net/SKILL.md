---
name: netrun-net
description: "The netrun Net class lifecycle and execution API: creating/starting/stopping a Net, inject_data() and NodeInfo injection helpers, the run_until_blocked/execute_epoch execution loop, execute_startable_epochs(), background execution with start_background/wait_until_done, and the full output queue API (get_output, try_get_output, flush_output_queue, ConsumedOutputPacket metadata)."
---

# The Net Class

`Net` is the main runtime class. It manages the simulation, pools, packet storage, and epoch execution.

## Creating a Net

```python
from netrun.core import Net, NetConfig

# From file
net = Net(NetConfig.from_file("main.netrun.json"))

# Class method shorthand
net = Net.from_file("main.netrun.json")
```

## Lifecycle

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

## Injecting Data

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

## Execution Loop

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

The execution loop pattern:
1. `run_until_blocked()` — moves packets along edges and creates startable epochs.
2. `get_startable_epochs()` — returns epochs ready to execute.
3. `execute_epoch()` — runs the node function for each epoch.
4. Repeat until no more startable epochs exist.

## Background Execution

For fire-and-forget scenarios:

```python
async with Net(config) as net:
    await net.start_background()
    net.inject_data("source", "in", [data])
    await net.wait_until_done()
    results = net.flush_output_queue("results")
```

Background mode runs the execution loop automatically. Use `pause()` / `resume()` to control it.

## Output Queues

Output queues capture packets from specified output ports. Configure in the net config:

```json
{
  "output_queues": {
    "results": {"ports": [["format", "out"]]},
    "debug":   {"ports": [["node_a", "debug"], ["node_b", "debug"]]}
  }
}
```

### Retrieving Results

```python
# Blocking wait (async)
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

### ConsumedOutputPacket Fields

| Field | Type | Description |
|-------|------|-------------|
| `value` | `Any` | The packet value |
| `from_node` | `str` | Source node name |
| `from_port` | `str` | Source port name |
| `timestamp` | `datetime` | When the packet was produced |
| `epoch_id` | `str` | The epoch that produced it |

## Sample Projects

- **00** (`sample_projects/00_basic_net_project/`): Basic execution loop, inject_data, output queues
- **01** (`sample_projects/01_thread_and_process_pools/`): Execution with different pool types
- **04** (`sample_projects/04_error_handling/`): Execution with error handling
- **07** (`sample_projects/07_run_to_targets/`): Targeted execution with run_to_targets
