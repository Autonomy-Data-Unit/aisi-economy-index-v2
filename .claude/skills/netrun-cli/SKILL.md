---
name: netrun-cli
description: "The netrun CLI reference: all commands (validate, info, structure, nodes, node, convert, factory-info), actions commands (list, run with --global and --timeout), recipes commands (list, run with --inputs-json and --output), common options (-c/--config auto-detection, --pretty/--compact), and netrun-ui launch modes (native, server, dev, foreground)."
---

# CLI Reference

The `netrun` CLI provides commands for inspecting and managing netrun projects. Use this to quickly get lightweight context about the net and other relevant info.

## Core Commands

| Command | Description |
|---------|-------------|
| `netrun validate [-c CONFIG]` | Validate a netrun config file |
| `netrun info [-c CONFIG]` | Summary statistics (node count, pools, factories, etc.) |
| `netrun structure [-c CONFIG]` | Output graph topology as JSON |
| `netrun nodes [-c CONFIG]` | List all nodes with port names |
| `netrun node NAME [-c CONFIG]` | Detailed info about a specific node |
| `netrun convert FILE [-o OUTPUT]` | Convert between JSON and TOML formats |
| `netrun factory-info FACTORY_PATH` | Inspect a factory module's parameters |

### Command Details

**`netrun validate`** — Parses and validates the config file, resolving factories and checking for errors (missing nodes, invalid edges, etc.). Returns exit code 0 on success.

**`netrun info`** — Prints a summary including: number of nodes, edges, pools, factories used, output queues, and whether caching is enabled.

**`netrun structure`** — Outputs the fully resolved graph topology as JSON, including all nodes (with ports), edges, and subgraphs flattened.

**`netrun nodes`** — Lists all nodes with their input and output port names. Useful for quick reference when writing edges or injection code.

**`netrun node NAME`** — Detailed info for a single node: factory, factory_args, all ports with types and slot specs, salvo conditions, execution_config, and extra metadata.

**`netrun convert`** — Converts between `.netrun.json` and `.netrun.toml` formats. Output format is inferred from the `-o` extension.

**`netrun factory-info`** — Inspects a factory module and reports its parameters. Shows `get_node_config` parameters (excluding `_net_config`) and their types/defaults. Useful for discovering what `factory_args` a factory accepts.

## Actions Commands

| Command | Description |
|---------|-------------|
| `netrun actions list [-c CONFIG] [-n NODE]` | List available actions |
| `netrun actions run ACTION_ID [NODE] [-c CONFIG] [-g] [-t TIMEOUT]` | Execute an action |

**`netrun actions list`** — Lists all project-level actions. Use `-n NODE` to also show node-level actions for that node.

**`netrun actions run`** — Executes an action by its ID. Provide a node name to run with that node's context (template variables resolved). Use `--global` / `-g` to run without a node context. Use `-t` / `--timeout` to set a timeout in seconds.

## Recipes Commands

| Command | Description |
|---------|-------------|
| `netrun recipes list [-c CONFIG]` | List available recipes |
| `netrun recipes run NAME [-c CONFIG] [-i INPUTS_JSON] [-o OUTPUT]` | Execute a recipe |

**`netrun recipes list`** — Lists all defined recipes with their descriptions.

**`netrun recipes run`** — Executes a recipe by name. Use `-i` to provide inputs as a JSON string (skips interactive prompts). Use `-o` to write the modified config to a file (default: stdout).

```bash
# Example: run recipe with inputs, write to file
netrun recipes run add_node -i '{"node_name": "transform", "pool": "threads"}' -o main.netrun.json
```

## Common Options

| Option | Description |
|--------|-------------|
| `-c, --config PATH` | Config file path. If omitted, auto-detects `.netrun.json` or `.netrun.toml` in the current directory. |
| `--pretty / --compact` | JSON output formatting (default: `--pretty`) |

## netrun-ui CLI

```bash
netrun-ui [FILE]                       # Native window (background)
netrun-ui FILE --fg                    # Native window (foreground, blocks)
netrun-ui --server                     # Browser mode (opens http://localhost:PORT)
netrun-ui --dev --fg                   # Dev mode with Vite hot reload
netrun-ui --dev --server               # Dev mode in browser
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

### Launch Modes

| Mode | Command | Description |
|------|---------|-------------|
| Native (background) | `netrun-ui` | Opens native window, returns control to terminal |
| Native (foreground) | `netrun-ui --fg` | Opens native window, blocks until closed |
| Server | `netrun-ui --server` | Browser-based, production build |
| Dev | `netrun-ui --dev --fg` | Native window with Vite hot reload |
| Dev server | `netrun-ui --dev --server` | Browser with Vite hot reload |
