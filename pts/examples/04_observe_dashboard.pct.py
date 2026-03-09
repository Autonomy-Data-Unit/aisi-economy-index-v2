# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Netrun Observer Dashboard
#
# This notebook demonstrates the `ai_index.utils.observe` module, which provides
# live observability for netrun pipelines via a FastAPI server and web dashboard.
#
# We'll set up a small pipeline, start the observer server alongside it, and show
# how to query status both from the dashboard UI and the Python client.

# %% [markdown]
# ## 1. Define a toy pipeline
#
# Two nodes: `generate` produces items, `process` consumes them with a small delay
# so we can observe epochs in flight.

# %%
import asyncio
import json
import sys
import time
from pathlib import Path

# Write node functions to a temp module that netrun workers can import
_tmp_dir = Path("/tmp/observe_example")
_tmp_dir.mkdir(exist_ok=True)

_nodes_code = '''
import time

def generate(seed: int, print) -> int:
    """Produce items from a seed value."""
    for i in range(5):
        print(f"generating item {i}")
    return seed * 10

def process(data: int, print) -> str:
    """Process an item with a small delay."""
    print(f"processing {data}...")
    time.sleep(0.5)
    print(f"done processing {data}")
    return f"result_{data}"
'''
(_tmp_dir / "_observe_example_nodes.py").write_text(_nodes_code)

# Make the module importable by netrun workers
if str(_tmp_dir) not in sys.path:
    sys.path.insert(0, str(_tmp_dir))

# %%
config_dict = {
    "output_queues": {"results": {"ports": [["process", "out"]]}},
    "graph": {
        "nodes": [
            {
                "name": "generate",
                "factory": "netrun.node_factories.from_function",
                "factory_args": {"func": "_observe_example_nodes.generate"},
            },
            {
                "name": "process",
                "factory": "netrun.node_factories.from_function",
                "factory_args": {"func": "_observe_example_nodes.process"},
            },
        ],
        "edges": [
            {
                "source_str": "generate.out",
                "target_str": "process.data",
            },
        ],
    },
}

config_path = _tmp_dir / "config.netrun.json"
config_path.write_text(json.dumps(config_dict, indent=2))

# %% [markdown]
# ## 2. Start the pipeline with the observer server

# %%
from netrun.core import Net, NetConfig
from ai_index.utils.observe import ObserveServer

config = NetConfig.from_file(str(config_path))
net = Net(config, run_source_nodes=False)

# %%
server = ObserveServer(net, port=8000)

# %% [markdown]
# Start the server — this is non-blocking, it runs as a background asyncio task.
# In a regular script you'd use `await server.start()`. In a notebook with
# `nest_asyncio`, we can do:

# %%
import nest_asyncio
nest_asyncio.apply()

await server.start()
print("Server running at http://127.0.0.1:8000")
print("Dashboard at  http://127.0.0.1:8000/dashboard")
print("API docs at   http://127.0.0.1:8000/docs")

# %% [markdown]
# ## 3. Use the Python client to query status
#
# The `NetObserverClient` gives you the same interface as `NetObserver`, but
# over HTTP — so it works from a separate process or machine.

# %%
from ai_index.utils.observe import NetObserverClient

client = NetObserverClient("http://127.0.0.1:8000")

status = client.get_status()
print("Net status:", status)

# %%
nodes = client.get_nodes()
for n in nodes:
    print(f"  {n.name:20s}  busy={n.is_busy}  epochs={n.epoch_count}  ports_in={n.in_ports}  ports_out={n.out_ports}")

# %% [markdown]
# ## 4. Run the pipeline and observe it
#
# Inject data into the `generate` node and run epochs. While it runs, the
# dashboard at http://127.0.0.1:8000/dashboard will update live.

# %%
async def run_pipeline():
    async with net:
        net.inject_data("generate", "seed", [1])

        made_progress = True
        while made_progress:
            made_progress, _ = await net.run_until_blocked()

        return net.flush_output_queue("results")

results = await run_pipeline()
print("Pipeline results:", results)

# %% [markdown]
# Check the observer again after the run:

# %%
status = client.get_status()
print(f"Total epochs: {status.total_epochs}")
print(f"Epochs by state: {status.epochs_by_state}")

# %%
for e in client.get_epochs():
    dur = f"{e.duration_seconds:.3f}s" if e.duration_seconds else "-"
    print(f"  {e.epoch_id[:8]}  {e.node_name:20s}  {e.state:10s}  {dur}")

# %%
print("Logs:")
for log in client.get_all_logs():
    print(f"  [{log.node_name}] {log.message}")

# %% [markdown]
# ## 5. Clean up

# %%
client.close()
await server.stop()
print("Server stopped.")

# %% [markdown]
# ## Usage in a real pipeline
#
# To add the observer to the actual AISI pipeline, add a few lines to
# `run_pipeline_async`:
#
# ```python
# from ai_index.utils.observe import ObserveServer
#
# async def run_pipeline_async(run_name=None):
#     ...
#     async with Net(config) as net:
#         server = ObserveServer(net, port=8000)
#         await server.start()
#
#         made_progress = True
#         while made_progress:
#             made_progress, _ = await net.run_until_blocked()
#
#         await server.stop()
# ```
#
# Then open http://localhost:8000/dashboard in your browser while the pipeline runs.
