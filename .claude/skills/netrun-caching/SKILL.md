---
name: netrun-caching
description: "Netrun caching and memoization: CacheConfig (enabled, version, storage_path, include_all_nodes, include_nodes, exclude_nodes, cache_what), cache modes (both/output/input), per-node NodeCacheConfig overrides, version-based invalidation, persistent storage, and the full cache API (cache_stats, get_cached_entries, get_cached_input_salvos, get_cached_output_salvos, get_cached_output_for_input, clear_cache, clear_node_cache, epoch.was_cache_hit)."
---

# Caching & Memoization

netrun can cache epoch inputs and outputs so repeated inputs skip execution entirely.

## Enabling Caching

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

You can also exclude specific nodes:

```json
{
  "cache": {
    "enabled": true,
    "include_all_nodes": true,
    "exclude_nodes": ["volatile_node"]
  }
}
```

## Cache Modes

| Mode | Description |
|------|-------------|
| `both` (default) | On hit: skip execution, replay cached output. On miss: execute and store both input and output. |
| `output` | Always execute. Record outputs for inspection. |
| `input` | Always execute. Record inputs for replay testing. |

## Per-Node Overrides

Override cache settings for individual nodes via `execution_config.cache`:

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

Non-null fields in `NodeCacheConfig` override the corresponding net-level `CacheConfig` fields.

## Version-Based Invalidation

Change the version number to invalidate all cached entries:

```json
{
  "cache": {
    "enabled": true,
    "version": 2
  }
}
```

Only entries matching the current version are used for cache hits. Old entries remain in storage but are ignored.

## Persistent Storage

By default, cache is stored in a temporary directory and lost between runs. Set `storage_path` to persist:

```json
{
  "cache": {
    "enabled": true,
    "storage_path": "./.cache/netrun"
  }
}
```

## CacheConfig Fields

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

## Cache API

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

### NodeInfo Cache Helpers

```python
node = net.nodes["my_node"]
node.is_cache_enabled   # bool
node.cache_stats        # Stats dict for this node
node.cached_entries     # All cached entries for this node
```

## Detecting Cache Hits

```python
for epoch_id, epoch in net.epochs.items():
    if epoch.was_cache_hit:
        print(f"{epoch.node_name}: served from cache")
```

## Sample Projects

- **08** (`sample_projects/08_caching/`): All caching features — CacheConfig, modes, version invalidation, persistent storage, cache API, NodeInfo helpers, `epoch.was_cache_hit`
