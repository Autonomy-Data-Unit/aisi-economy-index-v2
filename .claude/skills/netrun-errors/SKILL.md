---
name: netrun-errors
description: "Netrun error handling: automatic retries (retries, retry_wait, ctx.retry_count), on_node_failure callbacks (NodeFailureContext), ctx.cancel_epoch(), exception propagation control (propagate_exceptions, net.exception_queue, net.propagate_exceptions()), dead letter queue (net.dead_letter_queue, net.clear_dead_letter_queue()), timeouts (requires thread/process pool), and type checking (type_checking_enabled, PacketTypeMismatch)."
---

# Error Handling

## Retries

Configure automatic retries with optional wait time:

```json
{
  "execution_config": {
    "retries": 3,
    "retry_wait": 0.5
  }
}
```

Inside the node, `ctx.retry_count` tracks the current attempt (0 on first try):

```python
def flaky_node(data: str, ctx, print) -> str:
    if ctx.retry_count < 2:
        raise ValueError(f"Transient error (attempt {ctx.retry_count + 1})")
    print(f"Attempt {ctx.retry_count + 1}: success!")
    return data.upper()
```

Additional retry context available:
- `ctx.retry_timestamps` — `list[datetime]` of when each previous attempt occurred
- `ctx.retry_exceptions` — `list[Exception]` of exceptions from previous attempts

## On-Failure Callback

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

The `NodeFailureContext` provides:
- `ctx.epoch_id` — The epoch that failed
- `ctx.node_name` — The node name
- `ctx.exception` — The exception that was raised
- `ctx.retry_count` — Current retry attempt number
- `ctx.print(...)` — Print function for logging

## Epoch Cancellation

A node can cancel its own epoch, discarding all in-flight packets:

```python
def my_node(data: str, ctx) -> str:
    if data == "invalid":
        ctx.cancel_epoch()  # Raises EpochCancelled
    return data
```

Key differences from exceptions:
- Cancelled epochs are **not** retried
- Cancelled epochs do **not** trigger `on_node_failure`
- Cancelled epochs do **not** enter the dead letter queue
- All packets within the epoch are destroyed

## Exception Propagation

By default, exceptions propagate and stop the network. Set `propagate_exceptions: false` to queue them instead:

```json
{
  "execution_config": {
    "propagate_exceptions": false
  }
}
```

Or at the net level:

```json
{
  "propagate_exceptions": false
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

## Dead Letter Queue

Failed epochs (after exhausting all retries) are sent to the dead letter queue when `dead_letter_queue: true` (the default):

```python
for entry in net.dead_letter_queue:
    print(entry)

# Drain and return
entries = net.clear_dead_letter_queue()
```

Additional DLQ configuration:

```json
{
  "dead_letter_queue": true,
  "dead_letter_path": "./dlq.json",
  "dead_letter_callback": "nodes.on_dead_letter"
}
```

## Timeouts

Timeouts require a thread or process pool (the main pool runs in the event loop and cannot interrupt):

```json
{
  "execution_config": {
    "pools": ["threads"],
    "timeout": 5.0
  }
}
```

When a timeout occurs, the epoch is treated as a failure (subject to retries if configured).

## Type Checking

Enable at the net or node level to validate packet types against port annotations:

```json
{
  "type_checking_enabled": true
}
```

A type mismatch raises `PacketTypeMismatch`. Validation uses beartype and supports:
- Python built-in types (`int`, `str`, `float`, `bool`, etc.)
- Generic aliases (`list[int]`, `dict[str, Any]`)
- Custom classes

Disable for specific nodes:

```json
{
  "execution_config": {
    "type_checking_enabled": false
  }
}
```

## Sample Projects

- **04** (`sample_projects/04_error_handling/`): All error handling features — retries, `on_node_failure`, `ctx.cancel_epoch()`, type checking, `propagate_exceptions: false`, timeout, dead letter queue, `ctx.retry_count`, `ctx.retry_exceptions`
