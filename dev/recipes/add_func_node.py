"""Recipe: Add a function node.

Creates a factory node backed by netrun.node_factories.from_function
and generates a stub Python function in nodes.py.
"""
import copy
import subprocess
from pathlib import Path

def get_prompts(config):
    """Return prompts for user input before running the recipe."""
    return [
        {
            "name": "node_name",
            "label": "Node name",
            "type": "text",
            "default": "my_node",
        },
    ]


def run(config, inputs):
    """Add a function factory node to the graph."""
    result = copy.deepcopy(config)

    project_root_path = Path(config['project_root_path'])

    # Create a new node
    node_name = inputs.get("node_name", "my_function")
    in_port_names = ["in1", "in2"]
    out_port_names = ["out1", "out2"]
    func_import = f"ai_index.nodes.{node_name}.{node_name}"

    new_node = {
        "type": "netrunNode",
        "data": {
            "label": node_name,
            "nodeType": "factory",
            "factory": "netrun.node_factories.from_function",
            "factoryArgs": {"func": func_import},
            "inPorts": [{"name": name} for name in in_port_names],
            "outPorts": [{"name": name} for name in out_port_names],
        },
    }

    # Create a func notebook
    nb_path = project_root_path / f"nbs/ai_index/nodes/{node_name}.ipynb"
    print(f"Creating notebook at {nb_path}")
    subprocess.run([
        "nbl", "new",
        "--template", project_root_path / "dev/templates/func_node.pct.py.jinja",
        nb_path
    ])
    
    # Export it to the .py file
    print(f"Exporting notebook")
    subprocess.run([
        "nbl", "export", nb_path
    ])

    result.setdefault("nodes", []).append(new_node)
    return result
