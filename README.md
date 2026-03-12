# AISI Economy Index v2

A data pipeline for measuring AI exposure across the UK economy. Matches ~30M Adzuna job advertisements to 861 O\*NET occupations using embedding similarity and LLM filtering, then computes multi-dimensional AI exposure scores and aggregates them by geography.

## Pipeline overview

The pipeline has three stages, orchestrated as a 20-node DAG via [netrun](https://github.com/lukastk/netrun):

**Job ad processing** -- Fetch and sample ads, extract structured summaries via LLM, embed ads and O\*NET occupations, compute cosine similarity matches, then LLM-filter candidates down to 2-3 functional occupation matches per ad.

**O\*NET exposure scoring** -- Compute four independent AI exposure scores per occupation: humanness/presence (physical, emotional, creative), Felten AIOE (ability-application relatedness), and LLM-classified task exposure. These run in parallel with the ad processing branch.

**Index construction** -- Map occupation-level scores to individual ads via weighted averaging, then aggregate by Local Authority District.

[View interactive pipeline graph](https://htmlpreview.github.io/?https://github.com/Autonomy-Data-Unit/aisi-economy-index/blob/main/docs/ai_index_net.html)

```
fetch_adzuna -> sample_ads -> llm_summarise -> embed_ads -----> cosine_match -> llm_filter -> compute_job_ad_exposure -> aggregate_geo
                                                                     ^                              ^
fetch_onet -> prepare_onet_targets -> embed_onet -------------------|                              |
                                   -> score_presence  --|                                          |
                                   -> score_felten    --+--> join_scores -> combine_onet_exposure --+
                                   -> score_task_exposure
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env  # Configure ADZUNA_S3_PREFIX, OPENAI_API_KEY, etc.
```

## Running the pipeline

```bash
uv run run-pipeline [RUN_NAME]
```

Run definitions are in `config/run_defs.toml`. Available runs:

| Run | Description |
|-----|-------------|
| `baseline` | Full dataset, API models (default) |
| `test` | 10 ads, API models |
| `test_api` | 10 ads, OpenAI API (text-embedding-3-large, gpt-5.2) |
| `test_local` | 10 ads, local CPU models (bge-large, qwen-0.5b) |
| `test_sbatch` | 10 ads, Isambard HPC via Slurm (bge-large, qwen-7b) |

## Execution modes

Model execution is configured per-model in `config/embed_models.toml` and `config/llm_models.toml`:

- **api** -- OpenAI/Gemini API (no GPU needed)
- **local** -- Direct CUDA or CPU inference
- **sbatch** -- Serialise inputs, submit Slurm job to Isambard HPC, download results

## Project structure

```
config/              Configuration (pipeline graph, run definitions, model configs)
prompt_library/      LLM prompt templates
pts/                 Source notebooks (.pct.py) -- edit these
  ai_index/          Pipeline nodes and utilities
  isambard_utils/    HPC cluster interaction (SSH, Slurm, rsync)
  llm_runner/        Model inference (embeddings, LLM, cosine similarity)
src/                 Generated Python modules (do not edit)
store/               Data storage (inputs, pipeline intermediates, outputs)
```

Development uses [nblite](https://github.com/lukastk/nblite) for literate programming. Edit `.pct.py` files in `pts/`, then export:

```bash
uv run nbl export --reverse && uv run nbl export
```

## Testing

```bash
uv run pytest                # Unit tests
uv run nbl test              # Notebook execution tests
```
