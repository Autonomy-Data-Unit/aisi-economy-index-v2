---
name: netrun-actions
description: "Netrun actions and recipes: defining project-level and node-level actions in extra.ui.actions, template variables ($NODE_NAME, $NET_FILE_PATH, $NET_FILE_DIR, $PROJECT_ROOT, $DEFAULT_CMD, $NODE_CONFIG, custom ui.env), running actions via CLI. Recipes: defining in config, Python recipe files (run(config, inputs) and get_prompts(config)), prompt types (text/number/select/checkbox), and CLI execution."
---

# Actions & Recipes

## Actions

Actions are shell commands with template variable substitution. They are defined in the `extra.ui.actions` section and can be run via the CLI or the UI.

### Project-Level Actions

```json
{
  "graph": {
    "extra": {
      "ui": {
        "actions": [
          {
            "id": "action-show-info",
            "label": "Show Node Info",
            "command": "echo \"Node: $NODE_NAME\" && echo \"App: $APP_NAME\""
          }
        ],
        "env": {
          "APP_NAME": "my-app"
        }
      }
    }
  }
}
```

TOML equivalent:

```toml
[[graph.extra.ui.actions]]
id = "action-show-info"
label = "Show Node Info"
command = "echo \"Node: $NODE_NAME\" && echo \"App: $APP_NAME\""

[graph.extra.ui.env]
APP_NAME = "my-app"
```

### Node-Level Actions

```json
{
  "name": "my_node",
  "extra": {
    "ui": {
      "actions": [
        {
          "id": "action-test",
          "label": "Test This Node",
          "command": "echo \"Testing $NODE_NAME ($ROLE)\""
        }
      ],
      "env": {
        "ROLE": "processor"
      }
    }
  }
}
```

### Template Variables

| Variable | Description |
|----------|-------------|
| `$NODE_NAME` | Current node name |
| `$NET_FILE_PATH` | Full path to config file |
| `$NET_FILE_DIR` | Directory containing config file |
| `$PROJECT_ROOT` | Project root directory |
| `$DEFAULT_CMD` | Default command from `ui.defaultCmd` |
| `$NODE_CONFIG` | Node config as JSON string |
| Custom variables | From `ui.env` (project and node level) and `node_vars` |

### Running Actions via CLI

```bash
netrun actions list
netrun actions list --node my_node
netrun actions run action-show-info my_node
netrun actions run action-show-info --global
```

## Recipes

Recipes are Python scripts that transform the net configuration based on user inputs.

### Defining Recipes

```json
{
  "recipes": {
    "add_node": {
      "path": "./recipes/add_node.py",
      "description": "Add a new node to the graph"
    }
  }
}
```

### Recipe Python File

A recipe module must export two functions:

```python
# recipes/add_node.py

def get_prompts(config):
    """Return prompts for user input (optional).

    Args:
        config: The current net config as a dict

    Returns:
        List of prompt definitions
    """
    return [
        {"name": "node_name", "label": "Node name", "type": "text"},
        {"name": "pool", "label": "Pool", "type": "select", "options": ["main", "threads"], "default": "main"},
    ]

def run(config, inputs):
    """Transform the config based on user inputs.

    Args:
        config: The current net config as a dict
        inputs: Dict of user-provided values (keys match prompt names)

    Returns:
        Modified config dict
    """
    new_node = {
        "name": inputs["node_name"],
        "factory": "netrun.node_factories.from_function",
        "factory_args": {"func": f"nodes.{inputs['node_name']}"},
        "execution_config": {"pools": [inputs["pool"]]}
    }
    config["graph"]["nodes"].append(new_node)
    return config
```

### Prompt Types

| Type | Description | Extra Fields |
|------|-------------|--------------|
| `text` | Free-form text input | `default` (optional) |
| `number` | Numeric input | `default` (optional) |
| `select` | Single selection from list | `options` (required), `default` (optional) |
| `checkbox` | Multi-selection from list | `options` (required), `default` (optional) |

### Prompt Definition Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Key in the `inputs` dict passed to `run()` |
| `label` | Yes | Display label for the prompt |
| `type` | Yes | One of: `text`, `number`, `select`, `checkbox` |
| `options` | For select/checkbox | List of option strings |
| `default` | No | Default value |

### Running Recipes via CLI

```bash
netrun recipes list
netrun recipes run add_node -i '{"node_name": "transform", "pool": "threads"}' -o main.netrun.json
```

The `-i` flag provides inputs as JSON (skipping interactive prompts). The `-o` flag specifies the output file (default: stdout).

## Sample Projects

- **06** (`sample_projects/06_actions_and_recipes/`): Project-level and node-level actions, template variables, custom env, recipes with prompts
- **01** (`sample_projects/01_thread_and_process_pools/`): Recipe definitions
