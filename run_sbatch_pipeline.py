"""Run the pipeline via netrun Net.

All configuration is read from netrun.json node_vars (with env var overrides).
Caching is automatic — re-runs skip completed nodes.

Usage:
    EXECUTION_MODE=sbatch JOB_ADS_SAMPLE_RATE=0.001 python run_sbatch_pipeline.py
    LLM_BACKEND=vllm EXECUTION_MODE=sbatch python run_sbatch_pipeline.py
"""
import asyncio
import time
from importlib import resources
from pathlib import Path

from netrun.core import Net, NetConfig


async def main():
    config_path = resources.files("ai_index.assets") / "netrun.json"
    config = NetConfig.from_file(str(config_path))
    config.project_root_override = str(Path.cwd())

    t0 = time.time()

    async with Net(config) as net:
        made_progress = True
        while made_progress:
            made_progress, _ = await net.run_until_blocked()

        # Print captured node logs
        net.print_all_logs()

        # Epoch summary
        print()
        print("=" * 60)
        for eid, epoch in net.epochs.items():
            status = "CACHED" if epoch.was_cache_hit else epoch.state
            dt = ""
            if epoch.started_at and epoch.ended_at:
                dt = f" ({(epoch.ended_at - epoch.started_at).total_seconds():.1f}s)"
            print(f"  {epoch.node_name}: {status}{dt}")

        # Output queues
        results = net.flush_all_output_queues()
        for name, outputs in results.items():
            print(f"  output queue '{name}': {len(outputs)} packet(s)")

        total = time.time() - t0
        print(f"\nTotal: {total:.1f}s ({total/60:.1f} min)")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
