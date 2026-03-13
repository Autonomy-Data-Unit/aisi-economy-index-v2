# Validation Plan: Model Sensitivity Analysis

## Goal

Test how sensitive the job-ad-to-O*NET matching is to the choice of embedding and LLM model. The pipeline should produce roughly the same results regardless of model choice.

## Experimental Design

**Crossed design** (not full factorial):
- Fix embedding (e.g. `bge-large-sbatch`), vary all LLMs -> isolates LLM effect
- Fix LLM (e.g. `qwen-7b-sbatch`), vary all embeddings -> isolates embedding effect
- A handful of cross-combos for interaction effects

**Sample:** 100,000 ads, fixed `sample_seed=42` so every config gets the same ads.

**Estimated cost:** ~91 GPU-hours, ~23 NHR total for 21 runs (11 LLM + 10 embed). About 1-2 days wall clock.

## Statistics

1. **Top-1 agreement rate** - % of ads where two configs assign the same primary O*NET code.
2. **Top-K Jaccard similarity** - Average |A∩B| / |A∪B| over matched O*NET sets per ad.
3. **Rank-Biased Overlap (RBO)** - Like Jaccard but weights higher-ranked matches more.
4. **Per-stage decomposition** - Measure agreement at two checkpoints:
   - After `cosine_match` (candidate set, pre-filter)
   - After `llm_filter_candidates` (final set, post-filter)
   This tells us whether disagreements come from the embedding or the LLM.
5. **Stratified breakdown** - Agreement by Adzuna job category (~30 categories). Some sectors are inherently ambiguous.
6. **O*NET code distribution** - Chi-squared or KL divergence on marginal distribution of assigned codes. Catches systematic biases (e.g. one model over-assigns "Software Developer").

## Repo Structure

```
validation/
├── run_defs.toml          # Overrides: sample_n=100000, sample_seed=42, resume=false
├── run_validation.py      # Run pipeline for one (llm, embed) pair
├── run_all.py             # Crossed design orchestrator
├── analyze.py             # Read from store/pipeline/val__*/, compute stats
└── results/
    └── figures/           # Output plots
```

Pipeline results are read directly from `store/pipeline/val__{llm}__{embed}/`. No extraction needed.
