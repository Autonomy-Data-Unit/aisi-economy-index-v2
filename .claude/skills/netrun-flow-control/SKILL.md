---
name: netrun-flow-control
description: "Netrun flow control: custom salvo conditions, salvo condition terms (port/and/or/not), port state predicates (empty/full/equals/greater_than etc.), packet count modes (all/count N), finite port slot specs, and type checking with PacketTypeMismatch."
---

# Ports & Flow Control

## Port Slot Specs

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

## Salvo Conditions

Salvo conditions define **when** a node fires and **which** packets participate.

By default, the function factory generates:
- **Input salvo**: fires when all input ports have at least 1 packet (consumes 1 per non-list port, all for list ports).
- **Output salvo**: fires once, sending all output packets.

### Custom Input Salvo Conditions

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

### Salvo Condition Terms

Terms are boolean expressions over port states:

| Term Type | Description | JSON Example |
|-----------|-------------|--------------|
| `true` | Always true | `{"type": "true"}` |
| `false` | Always false | `{"type": "false"}` |
| `port` | Check a port's state | `{"type": "port", "port_name": "data", "state": {...}}` |
| `and` | All sub-terms must be true | `{"type": "and", "terms": [{...}, {...}]}` |
| `or` | Any sub-term must be true | `{"type": "or", "terms": [{...}, {...}]}` |
| `not` | Negate a sub-term | `{"type": "not", "term": {...}}` |

### Port State Predicates

| State | JSON | Description |
|-------|------|-------------|
| `empty` | `{"type": "empty"}` | Port has no packets |
| `non_empty` | `{"type": "non_empty"}` | Port has at least one packet |
| `full` | `{"type": "full"}` | Port is at capacity (finite slots only) |
| `non_full` | `{"type": "non_full"}` | Port is not at capacity |
| `equals` | `{"type": "equals", "value": 5}` | Packet count equals value |
| `less_than` | `{"type": "less_than", "value": 3}` | Count less than value |
| `greater_than` | `{"type": "greater_than", "value": 3}` | Count greater than value |
| `equals_or_less_than` | `{"type": "equals_or_less_than", "value": 5}` | Count <= value |
| `equals_or_greater_than` | `{"type": "equals_or_greater_than", "value": 3}` | Count >= value |

### Packet Count Modes

| Mode | JSON | Description |
|------|------|-------------|
| All | `{"type": "all"}` | Take all packets from the port |
| Count N | `{"type": "count", "count": 3}` | Take at most N packets |

### Combining Conditions (example: two-port AND condition)

```json
{
  "in_salvo_conditions": {
    "trigger": {
      "max_salvos": {"type": "finite", "max": 1},
      "ports": {
        "data": {"type": "count", "count": 1},
        "config": {"type": "count", "count": 1}
      },
      "term": {
        "type": "and",
        "terms": [
          {"type": "port", "port_name": "data", "state": {"type": "non_empty"}},
          {"type": "port", "port_name": "config", "state": {"type": "non_empty"}}
        ]
      }
    }
  }
}
```

## Type Checking

When `type_checking_enabled` is `true` (the default), netrun validates that packet values match port type annotations. A mismatch raises `PacketTypeMismatch`.

Supported types: Python types (`int`, `str`, `float`, etc.), generic aliases (`list[int]`), and custom classes. Validation uses beartype.

Enable/disable at net level or per-node:

```json
{
  "type_checking_enabled": true,
  "graph": {
    "nodes": [{
      "name": "relaxed_node",
      "execution_config": {
        "type_checking_enabled": false
      }
    }]
  }
}
```

## Sample Projects

- **05** (`sample_projects/05_advanced_flow_control/`): Custom salvo conditions, finite port slots, rate limiting, batch `list[T]` inputs
- **04** (`sample_projects/04_error_handling/`): Type checking (`type_checking_enabled`, `PacketTypeMismatch`)
