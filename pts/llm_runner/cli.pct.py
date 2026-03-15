# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CLI
#
# Command-line interface for llm_runner. Invoked as `python -m llm_runner <op>`.
#
# ```
# python -m llm_runner embed --inputs-dir /path --outputs-dir /path --config '{...}'
# python -m llm_runner llm_generate --inputs-dir /path --outputs-dir /path --config '{...}'
# python -m llm_runner cosine_topk --inputs-dir /path --outputs-dir /path --config '{...}'
# ```
#
# With manifest mode (for orchestration with per-key input directories):
# ```
# python -m llm_runner embed --manifest /path/manifest.json --outputs-dir /path --config '{...}'
# ```

# %%
#|default_exp cli

# %%
#|export
import argparse
import json
import sys
import traceback
from pathlib import Path

# %%
#|export
_OPERATIONS = {
    "embed": "llm_runner.embed:run_embeddings",
    "llm_generate": "llm_runner.llm:run_llm_generate",
    "cosine_topk": "llm_runner.cosine:run_cosine_topk",
    "rerank": "llm_runner.rerank:run_rerank",
    "rerank_pairs": "llm_runner.rerank:run_rerank_pairs",
}

# %%
#|export
def _load_inputs(args) -> dict:
    """Load inputs from either --inputs-dir or --manifest."""
    if args.manifest:
        from llm_runner.serialization import deserialize
        manifest_path = Path(args.manifest)
        with open(manifest_path) as f:
            manifest = json.load(f)
        # Manifest maps input keys to directories
        inputs = {}
        for key, remote_dir in manifest.items():
            key_data = deserialize(Path(remote_dir))
            # If the directory contains a single key matching the dir key, unwrap
            if list(key_data.keys()) == [key]:
                inputs[key] = key_data[key]
            else:
                inputs.update(key_data)
        return inputs
    else:
        from llm_runner.serialization import deserialize
        return deserialize(Path(args.inputs_dir))

# %%
#|export
def _run_operation(op_name: str, inputs: dict, config: dict) -> dict:
    """Import and run the operation function, return result as dict."""
    from llm_runner.embed import run_embeddings
    from llm_runner.llm import run_llm_generate
    from llm_runner.cosine import run_cosine_topk
    from llm_runner.rerank import run_rerank, run_rerank_pairs

    if op_name == "embed":
        result = run_embeddings(**inputs, **config)
        return {"embeddings": result}
    elif op_name == "llm_generate":
        result = run_llm_generate(**inputs, **config)
        return {"responses": result}
    elif op_name == "cosine_topk":
        result = run_cosine_topk(**inputs, **config)
        return result  # Already a dict with "indices" and "scores"
    elif op_name == "rerank":
        result = run_rerank(**inputs, **config)
        return result  # Already a dict with "indices" and "scores"
    elif op_name == "rerank_pairs":
        result = run_rerank_pairs(**inputs, **config)
        return {"scores": result}
    else:
        raise ValueError(f"Unknown operation: {op_name!r}")

# %%
#|export
def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="llm_runner",
        description="Run inference operations (embed, llm_generate, cosine_topk, rerank, rerank_pairs)",
    )
    parser.add_argument("operation", choices=list(_OPERATIONS.keys()),
                        help="Operation to run")
    parser.add_argument("--inputs-dir", help="Directory with serialized inputs")
    parser.add_argument("--manifest", help="JSON manifest mapping input keys to directories")
    parser.add_argument("--outputs-dir", required=True,
                        help="Directory for serialized outputs")
    parser.add_argument("--config", default="{}", help="JSON config for the operation")

    args = parser.parse_args(argv)

    if not args.inputs_dir and not args.manifest:
        parser.error("One of --inputs-dir or --manifest is required")

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    status = {"state": "RUNNING"}
    status_path = outputs_dir / "status.json"

    try:
        config = json.loads(args.config)
        inputs = _load_inputs(args)
        result = _run_operation(args.operation, inputs, config)

        from llm_runner.serialization import serialize
        serialize(result, outputs_dir)

        status = {"state": "COMPLETED"}
    except Exception as e:
        status = {
            "state": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        with open(status_path, "w") as f:
            json.dump(status, f)

# %%
#|export
#|eval: false
if __name__ == "__main__":
    main()
