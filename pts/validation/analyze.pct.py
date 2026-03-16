# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Validation Analysis Report
#
# Model sensitivity analysis for the job-ad-to-O\*NET matching pipeline.
# Compares validation runs across different LLM and embedding model choices,
# computing pairwise agreement statistics at both the cosine stage (pre-LLM-filter)
# and filtered stage (post-LLM-filter).

# %%
#|default_exp analyze
#|export_as_func true

# %%
#|set_func_signature
def analyze(run_def: str) -> "Path":
    """Run validation analysis for the given run definition."""
    ...

# %% [markdown]
# ## Discover available validation runs

# %%
#|top_export
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ai_index.const import validation_config_path, validation_results_path
from validation.run_validation import _make_run_name
from validation.utils import (
    discover_completed_runs,
    load_matches,
    matches_to_ranked_lists,
    load_ad_categories,
    build_arm_table,
    build_pairwise_matrix,
    print_arm_table,
    compute_pairwise,
    plot_arm_bars,
    plot_stage_comparison,
    plot_pairwise_matrix,
    plot_pairwise_heatmap,
    plot_stratified_dotplot,
    plot_onet_distribution,
    plot_kl_divergence_bars,
    save_fig,
)

# %%
all_completed = discover_completed_runs()
by_rd: dict[str, list[str]] = {}
for rn, rd, llm, embed, rerank in all_completed:
    by_rd.setdefault(rd, []).append(rn)

print("Available completed validation runs:")
for rd, runs in sorted(by_rd.items()):
    print(f"\n  {rd} ({len(runs)} runs):")
    for rn in runs:
        print(f"    {rn}")

# %%
run_def = "validation_5k"

# %% [markdown]
# ---
# # Analysis

# %% [markdown]
# ## Load config and discover runs

# %%
#|export
with open(validation_config_path, "rb") as f:
    config = tomllib.load(f)

completed = discover_completed_runs(run_def)

if not completed:
    raise ValueError(
        f"No completed validation runs found for run definition '{run_def}'.\n"
        f"Run: uv run validate-all {run_def}"
    )

rbo_p = config["rbo_p"]
fixed_llms = config["fixed_llms"]
fixed_embeddings = config["fixed_embeddings"]
fixed_rerankers = config["fixed_rerankers"]
completed_names = {rn for rn, _, _, _, _ in completed}

output_dir = validation_results_path / run_def
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / "figures"

# %%
print(f"Run definition: {run_def}")
print(f"Completed validation runs: {len(completed)}")
for rn, _, llm, embed, rerank in completed:
    print(f"  {rn}")

# %% [markdown]
# ## Arm 1: LLM Sensitivity
#
# For each fixed embedding model, compare how different LLMs affect the matching.
# Each reference LLM provides a baseline; other LLMs are compared against it.
# Shows agreement at both the cosine stage (before LLM filtering) and filtered
# stage (after).

# %%
#|export
all_llm_dfs = []

for fixed_embed in fixed_embeddings:
    print(f"\n{'#' * 70}")
    print(f"Fixed embedding: {fixed_embed}")
    print(f"{'#' * 70}")

    fixed = {"embed": fixed_embed, "rerank": None}

    for ref_llm in fixed_llms:
        ref_run = _make_run_name(run_def, ref_llm, fixed_embed)
        if ref_run not in completed_names:
            print(f"\n  Skipping reference LLM {ref_llm} (run not complete)")
            continue

        arm_df = build_arm_table(
            completed, config, run_def, "llm", fixed, ref_llm, rbo_p,
        )

        title = f"Vary LLM (embed={fixed_embed}, ref={ref_llm})"
        print_arm_table(arm_df, title)

        # Plots
        fig_suffix = f"{fixed_embed}__{ref_llm}"

        fig = plot_arm_bars(arm_df, title)
        if fig:
            save_fig(fig, figures_dir, f"llm_bars__{fig_suffix}")

        fig = plot_stage_comparison(arm_df, title)
        if fig:
            save_fig(fig, figures_dir, f"llm_stage__{fig_suffix}")

        fig = plot_kl_divergence_bars(arm_df, title)
        if fig:
            save_fig(fig, figures_dir, f"llm_kl__{fig_suffix}")

        # Collect for CSV
        if not arm_df.empty:
            arm_df = arm_df.copy()
            arm_df["fixed_embed"] = fixed_embed
            arm_df["ref_llm"] = ref_llm
            all_llm_dfs.append(arm_df)

    # Pairwise matrix: every LLM vs every other LLM (no reference bias)
    for stage in ("cosine", "filtered"):
        matrix = build_pairwise_matrix(completed, "llm", fixed, stage)
        if not matrix.empty:
            mean_agree = (matrix.sum(axis=1) - 1) / (len(matrix) - 1)
            print(f"\nPairwise Top-1 Agreement ({stage}, embed={fixed_embed})")
            print(matrix.to_string(float_format=lambda x: f"{x:.3f}"))
            print(f"\nMean agreement per LLM ({stage}):")
            for model, val in mean_agree.sort_values(ascending=False).items():
                print(f"  {model:<28s} {val:.4f}")

            fig = plot_pairwise_matrix(
                matrix, f"LLM Pairwise Top-1 ({stage}, embed={fixed_embed})")
            if fig:
                save_fig(fig, figures_dir, f"llm_pairwise__{fixed_embed}__{stage}")

# %% [markdown]
# ## Arm 2: Embedding Sensitivity
#
# For each fixed LLM, compare how different embedding models affect the matching.
# Each reference embedding provides a baseline; other embeddings are compared against it.

# %%
#|export
all_embed_dfs = []

for fixed_llm in fixed_llms:
    print(f"\n{'#' * 70}")
    print(f"Fixed LLM: {fixed_llm}")
    print(f"{'#' * 70}")

    fixed = {"llm": fixed_llm, "rerank": None}

    for ref_embed in fixed_embeddings:
        ref_run = _make_run_name(run_def, fixed_llm, ref_embed)
        if ref_run not in completed_names:
            print(f"\n  Skipping reference embedding {ref_embed} (run not complete)")
            continue

        arm_df = build_arm_table(
            completed, config, run_def, "embed", fixed, ref_embed, rbo_p,
        )

        title = f"Vary Embedding (llm={fixed_llm}, ref={ref_embed})"
        print_arm_table(arm_df, title)

        # Plots
        fig_suffix = f"{fixed_llm}__{ref_embed}"

        fig = plot_arm_bars(arm_df, title)
        if fig:
            save_fig(fig, figures_dir, f"embed_bars__{fig_suffix}")

        fig = plot_stage_comparison(arm_df, title)
        if fig:
            save_fig(fig, figures_dir, f"embed_stage__{fig_suffix}")

        fig = plot_kl_divergence_bars(arm_df, title)
        if fig:
            save_fig(fig, figures_dir, f"embed_kl__{fig_suffix}")

        # Collect for CSV
        if not arm_df.empty:
            arm_df = arm_df.copy()
            arm_df["fixed_llm"] = fixed_llm
            arm_df["ref_embed"] = ref_embed
            all_embed_dfs.append(arm_df)

    # Pairwise matrix: every embedding vs every other embedding
    for stage in ("cosine", "filtered"):
        matrix = build_pairwise_matrix(completed, "embed", fixed, stage)
        if not matrix.empty:
            mean_agree = (matrix.sum(axis=1) - 1) / (len(matrix) - 1)
            print(f"\nPairwise Top-1 Agreement ({stage}, llm={fixed_llm})")
            print(matrix.to_string(float_format=lambda x: f"{x:.3f}"))
            print(f"\nMean agreement per embedding ({stage}):")
            for model, val in mean_agree.sort_values(ascending=False).items():
                print(f"  {model:<28s} {val:.4f}")

            fig = plot_pairwise_matrix(
                matrix, f"Embedding Pairwise Top-1 ({stage}, llm={fixed_llm})")
            if fig:
                save_fig(fig, figures_dir, f"embed_pairwise__{fixed_llm}__{stage}")

# %% [markdown]
# ## Arm 3: Reranker Sensitivity
#
# For each fixed LLM + embedding pair, compare how different reranker models
# affect the final matching results. Since the reranker operates after cosine
# and LLM filter stages, only filtered-stage metrics are meaningful here.

# %%
#|export
all_rerank_dfs = []

for fixed_llm in fixed_llms:
    for fixed_embed in fixed_embeddings:
        print(f"\n{'#' * 70}")
        print(f"Fixed LLM: {fixed_llm}, Fixed embedding: {fixed_embed}")
        print(f"{'#' * 70}")

        fixed = {"llm": fixed_llm, "embed": fixed_embed}

        for ref_rerank in fixed_rerankers:
            ref_run = _make_run_name(run_def, fixed_llm, fixed_embed, ref_rerank)
            if ref_run not in completed_names:
                print(f"\n  Skipping reference reranker {ref_rerank} (run not complete)")
                continue

            arm_df = build_arm_table(
                completed, config, run_def, "rerank", fixed, ref_rerank, rbo_p,
            )

            title = f"Vary Reranker (llm={fixed_llm}, embed={fixed_embed}, ref={ref_rerank})"
            print_arm_table(arm_df, title)

            # Plots
            fig_suffix = f"{fixed_llm}__{fixed_embed}__{ref_rerank}"

            fig = plot_arm_bars(arm_df, title)
            if fig:
                save_fig(fig, figures_dir, f"rerank_bars__{fig_suffix}")

            fig = plot_kl_divergence_bars(arm_df, title)
            if fig:
                save_fig(fig, figures_dir, f"rerank_kl__{fig_suffix}")

            # Collect for CSV
            if not arm_df.empty:
                arm_df = arm_df.copy()
                arm_df["fixed_llm"] = fixed_llm
                arm_df["fixed_embed"] = fixed_embed
                arm_df["ref_rerank"] = ref_rerank
                all_rerank_dfs.append(arm_df)

        # Pairwise matrix: every reranker vs every other reranker
        matrix = build_pairwise_matrix(completed, "rerank", fixed, "filtered")
        if not matrix.empty:
            mean_agree = (matrix.sum(axis=1) - 1) / (len(matrix) - 1)
            print(f"\nPairwise Top-1 Agreement (filtered, llm={fixed_llm}, embed={fixed_embed})")
            print(matrix.to_string(float_format=lambda x: f"{x:.3f}"))
            print(f"\nMean agreement per reranker:")
            for model, val in mean_agree.sort_values(ascending=False).items():
                print(f"  {model:<28s} {val:.4f}")

            fig = plot_pairwise_matrix(
                matrix, f"Reranker Pairwise Top-1 (llm={fixed_llm}, embed={fixed_embed})")
            if fig:
                save_fig(fig, figures_dir, f"rerank_pairwise__{fixed_llm}__{fixed_embed}")

# %% [markdown]
# ## Pairwise heatmap
#
# All-vs-all top-1 agreement across every completed run.

# %%
#|export
fig = plot_pairwise_heatmap(completed)
if fig:
    save_fig(fig, figures_dir, "pairwise_heatmap")

# %% [markdown]
# ## Representative pair diagnostics
#
# Pick two runs and show per-category agreement and O\*NET distribution comparison.

# %%
#|export
if len(completed) >= 2:
    run_a, _, llm_a, embed_a, rerank_a = completed[0]
    run_b, _, llm_b, embed_b, rerank_b = completed[1]

    sets_a = matches_to_ranked_lists(load_matches(run_a, "filtered"))
    sets_b = matches_to_ranked_lists(load_matches(run_b, "filtered"))

    common_ids = list(set(sets_a) & set(sets_b))
    if common_ids:
        categories = load_ad_categories(common_ids)
        fig = plot_stratified_dotplot(
            sets_a, sets_b, categories,
            f"{run_a} vs {run_b}",
        )
        if fig:
            save_fig(fig, figures_dir, "stratified_dotplot")

    fig = plot_onet_distribution(
        sets_a, sets_b,
        ref_label=f"{llm_a}/{embed_a}",
        other_label=f"{llm_b}/{embed_b}",
    )
    if fig:
        save_fig(fig, figures_dir, "onet_distribution")

# %% [markdown]
# ## Display saved plots

# %%
from IPython.display import Image, display

for fig_path in sorted(figures_dir.glob("*.png")):
    print(f"\n{fig_path.stem}")
    display(Image(filename=str(fig_path)))

# %% [markdown]
# ## Save combined CSVs

# %%
#|export
if all_llm_dfs:
    llm_path = output_dir / "llm_sensitivity.csv"
    pd.concat(all_llm_dfs, ignore_index=True).to_csv(llm_path, index=False)
    print(f"\nSaved: {llm_path}")
if all_embed_dfs:
    embed_path = output_dir / "embed_sensitivity.csv"
    pd.concat(all_embed_dfs, ignore_index=True).to_csv(embed_path, index=False)
    print(f"Saved: {embed_path}")
if all_rerank_dfs:
    rerank_path = output_dir / "rerank_sensitivity.csv"
    pd.concat(all_rerank_dfs, ignore_index=True).to_csv(rerank_path, index=False)
    print(f"Saved: {rerank_path}")

print(f"\nAll results saved to {output_dir}")

output_dir #|func_return_line
