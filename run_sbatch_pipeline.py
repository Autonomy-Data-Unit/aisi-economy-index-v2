"""Run the matching pipeline with real data via sbatch on Isambard.

Usage:
    EXECUTION_MODE=sbatch JOB_ADS_SAMPLE_RATE=0.001 python run_sbatch_pipeline.py
"""
import time
import os
from types import SimpleNamespace

t0_total = time.time()

node_vars = {
    "execution_mode": os.environ.get("EXECUTION_MODE", "local"),
    "embedding_model": os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
    "embedding_dtype": "float16",
    "embed_onet_batch_size": 64,
    "embed_job_ads_batch_size": 512,
    "embed_job_ads_chunk_size": 20000,
    "cosine_top_k": 5,
    "cosine_batch_size": 16384,
    "llm_model": os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    "llm_dtype": "float16",
    "llm_batch_size": 128,
    "llm_max_new_tokens": 60,
    "llm_max_keep": 5,
    "llm_backend": os.environ.get("LLM_BACKEND", "transformers"),
    "job_ads_sample_rate": float(os.environ.get("JOB_ADS_SAMPLE_RATE", "1.0")),
    "job_ads_random_seed": os.environ.get("JOB_ADS_RANDOM_SEED", ""),
}
ctx = SimpleNamespace(vars=node_vars)
mode = node_vars["execution_mode"]

print("=" * 60)
print(f"STEP 1: Loading source data (local)")
print("=" * 60)
t0 = time.time()

from ai_index.nodes.fetch_onet import fetch_onet
onet_result = fetch_onet(ctx, print)
onet_tables = onet_result["onet_tables"]

from ai_index.nodes.build_onet_descriptions import build_onet_descriptions
desc_result = build_onet_descriptions(onet_tables, ctx, print)
descriptions = desc_result["descriptions"]
print(f"  => {len(descriptions['soc_codes'])} occupations")

from ai_index.nodes.load_job_ads import load_job_ads
ads_result = load_job_ads(ctx, print)
job_ads = ads_result["job_ads"]
print(f"  => {len(job_ads['job_ids'])} job ads loaded")
print(f"  Source data loaded in {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"STEP 2: Embed O*NET ({mode} -> Isambard GPU)")
print("=" * 60)
t0 = time.time()
from ai_index.nodes.embed_onet import embed_onet
eo_result = embed_onet(descriptions, ctx, print)
onet_emb = eo_result["onet_embeddings"]
print(f"  => role={onet_emb['role_embeddings'].shape}, task={onet_emb['task_embeddings'].shape}")
print(f"  embed_onet done in {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"STEP 3: Embed job ads ({mode} -> Isambard GPU)")
print("=" * 60)
t0 = time.time()
from ai_index.nodes.embed_job_ads import embed_job_ads
eja_result = embed_job_ads(job_ads, ctx, print)
job_emb = eja_result["job_ad_embeddings"]
print(f"  => role={job_emb['role_embeddings'].shape}, task={job_emb['task_embeddings'].shape}, ids={len(job_emb['job_ids'])}")
print(f"  embed_job_ads done in {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"STEP 4: Cosine similarity ({mode} -> Isambard GPU)")
print("=" * 60)
t0 = time.time()
from ai_index.nodes.compute_cosine_similarity import compute_cosine_similarity
cs_result = compute_cosine_similarity(descriptions, onet_emb, job_emb, ctx, print)
candidates = cs_result["candidates"]
print(f"  => {len(candidates['job_ids'])} jobs, top cands: {candidates['candidates'][0][:3]}")
print(f"  cosine_similarity done in {time.time()-t0:.1f}s")

print()
print("=" * 60)
print(f"STEP 5: LLM filter ({mode} -> Isambard GPU)")
print("=" * 60)
t0 = time.time()
from ai_index.nodes.llm_filter_candidates import llm_filter_candidates
llm_result = llm_filter_candidates(candidates, job_ads, ctx, print)
weighted = llm_result["weighted_codes"]
print(f"  => {len(weighted['job_ids'])} jobs with weighted codes")
print(f"  sample weights[0] = {weighted['weights'][0]}")
print(f"  llm_filter done in {time.time()-t0:.1f}s")

total_time = time.time() - t0_total
print()
print("=" * 60)
print(f"ALL 4 MATCHING PIPELINE NODES COMPLETED (mode={mode})")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print("=" * 60)
