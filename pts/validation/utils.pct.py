# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # validation.utils
#
# Utility functions for validation analysis: run discovery, data loading,
# and pairwise agreement metrics.

# %%
#|default_exp utils

# %%
#|export
from pathlib import Path

import numpy as np
import pandas as pd

from ai_index.const import pipeline_store_path


def build_model_name_lookup() -> dict[str, str]:
    """Build a mapping from model config keys (e.g. 'qwen-7b-sbatch') to short display names.

    Reads all three model config files (llm_models.toml, embed_models.toml, rerank_models.toml),
    extracts the HuggingFace model identifier, and returns the part after the '/'.
    """
    import tomllib
    from ai_index.const import config_path

    lookup = {}
    for config_file in ["llm_models.toml", "embed_models.toml", "rerank_models.toml"]:
        path = config_path / config_file
        if not path.exists():
            continue
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        for key, entry in cfg.get("models", {}).items():
            hf_name = entry.get("model", key)
            short = hf_name.split("/")[-1] if "/" in hf_name else hf_name
            lookup[key] = short
    return lookup


def build_model_info_table(keys: list[str], lookup: dict[str, str]) -> pd.DataFrame:
    """Build a display table of model keys, short names, and full HF identifiers.

    Args:
        keys: Model config keys to include.
        lookup: Output of build_model_name_lookup().
    """
    import tomllib
    from ai_index.const import config_path

    # Build full HF name lookup
    full_names = {}
    for config_file in ["llm_models.toml", "embed_models.toml", "rerank_models.toml"]:
        path = config_path / config_file
        if not path.exists():
            continue
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        for key, entry in cfg.get("models", {}).items():
            full_names[key] = entry.get("model", key)

    rows = []
    for key in keys:
        rows.append({
            "key": key,
            "model": lookup.get(key, key),
            "hf_name": full_names.get(key, ""),
        })
    return pd.DataFrame(rows)


def notebook_to_report(notebook_path: str | Path, output_dir: str | Path, stem: str) -> Path:
    """Convert an executed notebook to a Markdown report (and optionally PDF).

    Code cells are omitted. Images from cell outputs are embedded as inline
    base64 data URIs. The markdown file is self-contained (no external image files).

    Args:
        notebook_path: Path to the .ipynb file (should already be executed with outputs).
        output_dir: Directory to write the report files.
        stem: Base filename without extension (e.g. "validation_5k__arm_1").

    Returns:
        Path to the generated markdown file.
    """
    import json, base64, re, subprocess

    notebook_path = Path(notebook_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(notebook_path) as f:
        nb = json.load(f)

    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            parts.append("".join(cell["source"]))
            parts.append("")

        elif cell["cell_type"] == "code":
            # Skip code, but include outputs
            for output in cell.get("outputs", []):
                otype = output.get("output_type", "")

                # Display data (styled tables, images, markdown)
                if otype in ("display_data", "execute_result"):
                    data = output.get("data", {})
                    # Prefer HTML (rendered tables with styling)
                    if "text/html" in data:
                        html = "".join(data["text/html"])
                        parts.append(html)
                        parts.append("")
                    # Images
                    elif "image/png" in data:
                        b64 = "".join(data["image/png"]).strip()
                        parts.append(f"![](data:image/png;base64,{b64})")
                        parts.append("")
                    # Plain text fallback
                    elif "text/plain" in data:
                        text = "".join(data["text/plain"])
                        # Skip repr-style outputs like '<pandas.io.formats...>'
                        if not text.startswith("<"):
                            parts.append(f"```\n{text}\n```")
                            parts.append("")

                # Stream output (print statements)
                elif otype == "stream":
                    text = "".join(output.get("text", []))
                    if text.strip():
                        parts.append(f"```\n{text.rstrip()}\n```")
                        parts.append("")

    md_content = "\n".join(parts)

    # Also embed any matplotlib figure outputs that nbconvert saved as attachments
    # (cell attachments use a different format)
    for cell in nb["cells"]:
        for att_name, att_data in cell.get("attachments", {}).items():
            for mime, b64 in att_data.items():
                if mime.startswith("image/"):
                    old_ref = f"attachment:{att_name}"
                    new_ref = f"data:{mime};base64,{b64}"
                    md_content = md_content.replace(old_ref, new_ref)

    md_path = output_dir / f"{stem}.md"
    md_path.write_text(md_content)

    return md_path



def discover_completed_runs(run_def: str | None = None) -> list[tuple[str, str, str, str, str]]:
    """Scan store/pipeline/ for completed validation runs.

    A run is complete if it has an exposure_meta.json (written last by the pipeline).

    Returns list of (run_name, run_def, llm_key, embed_key, rerank_key) tuples.
    All runs have an explicit reranker key (5-part naming).
    """
    runs = []
    if not pipeline_store_path.exists():
        return runs

    for run_dir in sorted(pipeline_store_path.iterdir()):
        if not run_dir.name.startswith("val__"):
            continue
        meta = run_dir / "compute_job_ad_exposure" / "exposure_meta.json"
        if not meta.exists():
            continue
        parts = run_dir.name.split("__")
        if len(parts) != 5:
            continue
        _, rd, llm, embed, rerank = parts
        if run_def is not None and rd != run_def:
            continue
        runs.append((run_dir.name, rd, llm, embed, rerank))

    return runs


def load_parquet(run_name: str, node: str, filename: str) -> pd.DataFrame:
    """Load a parquet file from a pipeline run's node output directory."""
    return pd.read_parquet(pipeline_store_path / run_name / node / filename)

# %%
#|export
def pairwise_jaccard(sets_a: dict, sets_b: dict, common_keys: list) -> float:
    """Mean Jaccard similarity of set-valued dicts over common keys."""
    jaccards = []
    for k in common_keys:
        a = sets_a.get(k, set())
        b = sets_b.get(k, set())
        if not a and not b:
            continue
        jaccards.append(len(a & b) / len(a | b))
    return np.mean(jaccards) if jaccards else 0.0


def pairwise_weighted_jaccard(scores_a: dict, scores_b: dict, common_keys: list) -> float:
    """Mean weighted Jaccard (Ruzicka similarity) over common keys.

    Each value in scores_a/scores_b is a dict mapping candidate codes to scores.
    For each key: weighted intersection = sum(min(s_a, s_b)), weighted union = sum(max(s_a, s_b)).
    """
    wjs = []
    for k in common_keys:
        sa = scores_a.get(k, {})
        sb = scores_b.get(k, {})
        all_codes = set(sa.keys()) | set(sb.keys())
        if not all_codes:
            continue
        w_inter = sum(min(sa.get(c, 0.0), sb.get(c, 0.0)) for c in all_codes)
        w_union = sum(max(sa.get(c, 0.0), sb.get(c, 0.0)) for c in all_codes)
        wjs.append(w_inter / w_union if w_union > 0 else 0.0)
    return np.mean(wjs) if wjs else 0.0


def pairwise_spearman(scores_a: dict, scores_b: dict, common_keys: list) -> float:
    """Mean Spearman rank correlation on shared candidates over common keys.

    Only considers keys where at least 3 candidates are shared (needed for
    a meaningful rank correlation).
    """
    from scipy.stats import spearmanr

    correlations = []
    for k in common_keys:
        sa = scores_a.get(k, {})
        sb = scores_b.get(k, {})
        shared = set(sa.keys()) & set(sb.keys())
        if len(shared) < 3:
            continue
        vals_a = [sa[c] for c in shared]
        vals_b = [sb[c] for c in shared]
        rho, _ = spearmanr(vals_a, vals_b)
        if not np.isnan(rho):
            correlations.append(rho)
    return np.mean(correlations) if correlations else np.nan


def pairwise_top1(top1_a: dict, top1_b: dict, common_keys: list) -> float:
    """Fraction of common keys where two dicts have the same value."""
    agrees = sum(1 for k in common_keys if top1_a.get(k) == top1_b.get(k))
    return agrees / len(common_keys) if common_keys else 0.0


def pairwise_correlation_matrix(names: list[str], vectors: dict[str, np.ndarray], method: str = "pearson") -> pd.DataFrame:
    """Build an NxN correlation matrix from aligned score vectors.

    Args:
        names: Model names.
        vectors: Dict mapping model name to a 1-D array of scores (same length, same ad ordering).
        method: "pearson" or "spearman".
    """
    from scipy.stats import pearsonr, spearmanr

    fn = pearsonr if method == "pearson" else spearmanr
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            r, _ = fn(vectors[names[i]], vectors[names[j]])
            matrix[i, j] = r
            matrix[j, i] = r
    return pd.DataFrame(matrix, index=names, columns=names)


def build_pairwise_matrix(names: list[str], values: dict, common_keys: list, metric_fn) -> pd.DataFrame:
    """Build an NxN pairwise matrix by applying metric_fn(values[a], values[b], common_keys)."""
    n = len(names)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            score = metric_fn(values[names[i]], values[names[j]], common_keys)
            matrix[i, j] = score
            matrix[j, i] = score
    return pd.DataFrame(matrix, index=names, columns=names)


def upper_tri_stats(matrix: np.ndarray) -> dict:
    """Compute summary stats from the upper triangle of a square matrix."""
    n = matrix.shape[0]
    vals = matrix[np.triu_indices(n, k=1)]
    return {
        "mean": vals.mean(),
        "median": np.median(vals),
        "min": vals.min(),
        "max": vals.max(),
        "std": vals.std(),
    }


def best_subsets(names: list[str], matrix: np.ndarray, metric_fn_matrix=None) -> list[tuple[int, float, list[str]]]:
    """Find best model subsets by mean pairwise score from a matrix.

    Returns list of (k, score, subset_names) from k=len(names) down to k=2.
    """
    from itertools import combinations

    n = len(names)
    results = []
    for k in range(n, 1, -1):
        best_score = -1.0
        best_subset = None
        for subset in combinations(range(n), k):
            pairs = list(combinations(subset, 2))
            score = np.mean([matrix[i, j] for i, j in pairs])
            if score > best_score:
                best_score = score
                best_subset = subset
        subset_names = [names[i] for i in best_subset]
        results.append((k, best_score, subset_names))
    return results

# %%
#|export
def generate_reports_main():
    """CLI entry point: execute all analyze_* notebooks and generate markdown/PDF reports."""
    import os, sys
    import subprocess
    from ai_index.const import reports_path, repo_root

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print("Usage: uv run generate-reports <run_def_name>", file=sys.stderr)
        sys.exit(1)

    run_def = args[0]
    nbs_dir = repo_root / "nbs" / "validation"
    analyze_nbs = sorted(nbs_dir.glob("analyze_*.ipynb"))

    if not analyze_nbs:
        print(f"No analyze_* notebooks found in {nbs_dir}", file=sys.stderr)
        sys.exit(1)

    reports_path.mkdir(parents=True, exist_ok=True)

    for nb_path in analyze_nbs:
        # Derive report stem: analyze_arm1.ipynb -> {run_def}__arm_1
        arm_name = nb_path.stem.replace("analyze_", "")
        stem = f"{run_def}__{arm_name}"

        print(f"\n{'=' * 60}")
        print(f"Executing: {nb_path.name}")
        print(f"{'=' * 60}")

        # Execute the notebook in place, passing run_def via environment variable
        env = {**os.environ, "VALIDATION_RUN_DEF": run_def}
        result = subprocess.run(
            ["uv", "run", "jupyter", "nbconvert",
             "--to", "notebook", "--execute", "--inplace",
             str(nb_path)],
            capture_output=True, text=True, timeout=600, env=env,
        )
        if result.returncode != 0:
            print(f"  ERROR executing {nb_path.name}:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            continue

        # Convert executed notebook to markdown report
        md_path = notebook_to_report(nb_path, reports_path, stem)
        print(f"  Markdown: {md_path}")
