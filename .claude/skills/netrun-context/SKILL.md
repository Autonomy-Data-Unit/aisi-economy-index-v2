---
name: netrun-context
description: "The netrun NodeExecutionContext (ctx) API available inside node functions: epoch_id, node_name, retry_count/retry_timestamps/retry_exceptions, ctx.vars, packet operations (create_packet, create_packet_from_value_func, consume_packet, load_output_port, send_output_salvo), print capture, and ctx.cancel_epoch()."
---

# Node Execution Context

The `ctx` parameter provides access to the execution context inside a node function.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `ctx.epoch_id` | `str` | Unique epoch identifier |
| `ctx.node_name` | `str` | Name of the current node |
| `ctx.retry_count` | `int` | Current retry attempt (0 on first try) |
| `ctx.retry_timestamps` | `list[datetime]` | Timestamps of previous retry attempts |
| `ctx.retry_exceptions` | `list[Exception]` | Exceptions from previous retries |
| `ctx.vars` | `dict[str, Any]` | Resolved node variables (see netrun-execution-config skill) |

## Packet Operations

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

### Manual Output Mode

When `manual_output=True` in factory_args, the function must manage output entirely via `ctx`:

```json
{
  "name": "manual_node",
  "factory": "netrun.node_factories.from_function",
  "factory_args": {
    "func": "nodes.manual_processor",
    "manual_output": true
  }
}
```

```python
def manual_processor(data: str, ctx):
    # Process and create output manually
    result = data.upper()
    packet_id = ctx.create_packet(result)
    ctx.load_output_port("out", packet_id)
    ctx.send_output_salvo("send")
    # Must return None in manual mode
```

## Print Capture

The `print` parameter provides a captured print function. Output is logged with timestamps and appears in epoch logs:

```python
def my_node(data: str, print):
    print(f"Processing: {data}")   # Logged with timestamp
    print("Step 1 done")           # Appears in epoch logs
```

Print behavior is controlled by execution config:
- `capture_prints` (default `true`) — Whether to capture at all
- `print_flush_interval` (default `0.1`) — How often the buffer flushes (seconds)
- `print_buffer_max_size` (default `null`) — Max entries in buffer
- `print_echo_stdout` (default `false`) — Also print to real stdout

## Cancel Epoch

A node can cancel its own epoch. This raises `EpochCancelled` and discards all in-flight packets for the epoch:

```python
def my_node(data: str, ctx) -> str:
    if data == "invalid":
        ctx.cancel_epoch()  # Raises EpochCancelled, discards all in-flight packets
    return data.upper()
```

Cancellation is different from raising an exception:
- Cancelled epochs are not retried
- Cancelled epochs do not trigger `on_node_failure`
- Cancelled epochs do not enter the dead letter queue
- All packets within the epoch are destroyed

## Sample Projects

- **01** (`sample_projects/01_thread_and_process_pools/`): `ctx.vars` access
- **04** (`sample_projects/04_error_handling/`): `ctx.retry_count`, `ctx.retry_exceptions`, `ctx.cancel_epoch()`
- **08** (`sample_projects/08_caching/`): `ctx` usage with caching
