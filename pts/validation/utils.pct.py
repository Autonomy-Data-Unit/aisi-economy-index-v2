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
