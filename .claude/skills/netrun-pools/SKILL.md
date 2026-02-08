---
name: netrun-pools
description: "Netrun worker pool configuration: main, thread, multiprocess, and remote pool types with all options (num_workers, num_processes, threads_per_process, url, worker_name). Pool assignment via execution_config.pools, allocation methods (round-robin, random, least-busy), and remote pool server setup with Net.serve_pool()."
---

# Worker Pools

Pools determine **where** node functions execute. netrun supports four pool types.

## Pool Types

| Type | Description | Use Case |
|------|-------------|----------|
| `main` | Single async worker in the main event loop | I/O-bound work, default |
| `thread` | Multiple worker threads in the same process | Blocking I/O, timeouts |
| `multiprocess` | Separate subprocesses with worker threads | CPU-bound work |
| `remote` | Workers on remote machines via WebSocket | Distributed execution |

## Configuration

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

TOML equivalent:

```toml
[pools.main.spec]
type = "main"

[pools.threads.spec]
type = "thread"
num_workers = 4

[pools.processes.spec]
type = "multiprocess"
num_processes = 2
threads_per_process = 2

[pools.remote.spec]
type = "remote"
url = "ws://192.168.1.100:8765"
worker_name = "execution_manager"
num_processes = 1
threads_per_process = 1
```

If no pools are defined, a default `"main"` pool is created automatically.

### Pool-Level Settings

Each pool config also supports:

| Field | Default | Description |
|-------|---------|-------------|
| `print_flush_interval` | `0.1` | Print buffer flush interval (seconds) |
| `capture_prints` | `true` | Capture worker print output |

## Assigning Nodes to Pools

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

## Allocation Methods

| Method | Description |
|--------|-------------|
| `round-robin` | Cycles through pools in order (default) |
| `random` | Randomly selects a pool |
| `least-busy` | Picks the pool with the fewest running tasks |

Set the default at net level:

```json
{
  "default_pool_allocation_method": "round-robin"
}
```

Override per node:

```json
{
  "execution_config": {
    "pool_allocation_method": "least-busy"
  }
}
```

## Remote Pool Server

Start a remote pool server from your net config:

```python
server_ctx = Net.serve_pool(config, host="0.0.0.0", port=8765)
async with server_ctx:
    # Server is running, clients can connect
    await asyncio.Future()  # Run forever
```

The remote pool server hosts the execution environment. Clients connect via WebSocket using the `remote` pool type configuration. The `worker_name` in the client config must match the registered worker name on the server (typically `"execution_manager"`).

### Typical Remote Setup

1. **Server machine**: Run a script that calls `Net.serve_pool()` with the same config
2. **Client machine**: Define a `remote` pool pointing to the server's address
3. Nodes assigned to the remote pool execute on the server machine

## Pool Type Selection Guide

- **`main`**: Use for lightweight I/O-bound operations. No parallelism. Cannot use `timeout`.
- **`thread`**: Use when you need timeouts or concurrent blocking I/O. Shares GIL.
- **`multiprocess`**: Use for CPU-bound work. Full parallelism. Higher overhead for data transfer.
- **`remote`**: Use for distributed execution across machines. Requires network setup.

## Sample Projects

- **00** (`sample_projects/00_basic_net_project/`): Default main pool
- **01** (`sample_projects/01_thread_and_process_pools/`): All pool types, allocation methods
- **02** (`sample_projects/02_remote_deployment/`): Remote pool server, cloud deployment, SSH tunneling
- **04** (`sample_projects/04_error_handling/`): Thread pool for timeouts
