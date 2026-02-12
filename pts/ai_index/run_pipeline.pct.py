# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run Pipeline
#
# CLI entry point for running the AISI Economy Index pipeline end-to-end.

# %%
#|default_exp run_pipeline

# %%
#|export
import asyncio
from importlib import resources
from pathlib import Path
from netrun.core import Net, NetConfig

# %%
#|export
async def run_pipeline_async():
    """Load and run the full pipeline, returning output queue results."""
    config_path = resources.files("ai_index.assets") / "netrun.json"
    config = NetConfig.from_file(str(config_path))

    async with Net(config) as net:
        # Execute source nodes (no input ports, must be triggered explicitly)
        source_nodes = [
            name for name, node in net.nodes.items()
            if not node.in_port_names
        ]
        for name in source_nodes:
            await net.execute_node(name)

        # Run the rest of the pipeline
        made_progress = True
        while made_progress:
            made_progress, _ = await net.run_until_blocked()

        results = net.flush_all_output_queues()
        for queue_name, outputs in results.items():
            print(f"\n=== Output queue: {queue_name} ({len(outputs)} packet(s)) ===")
            for i, output in enumerate(outputs):
                print(f"  [{i}] {output}")

    return results

# %%
#|export
def main():
    """Sync entry point for the run-pipeline CLI command."""
    asyncio.run(run_pipeline_async())
