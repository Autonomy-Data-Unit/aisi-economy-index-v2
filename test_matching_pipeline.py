"""Quick synthetic test of the matching pipeline logic on CPU.

Creates fake upstream data, mocks GPU model calls, and runs each
matching node function to validate data contracts and logic.

Supports execution modes via EXECUTION_MODE env var:
  - local (default): mocks torch, tests GPU code paths with fake tensors
  - api: no torch mock needed, tests numpy cosine sim + mocked adulib LLM
"""
import os
import sys
import numpy as np
from types import SimpleNamespace

EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "local")

# ---------------------------------------------------------------------------
# 0. Install mocks BEFORE any node imports
# ---------------------------------------------------------------------------
if EXECUTION_MODE != "api":
    # Mock torch for local/deploy modes (no GPU available in test)
    class FakeTensor:
        def __init__(self, data):
            self.data = np.asarray(data)
        def __matmul__(self, other):
            return FakeTensor(self.data @ other.data)
        @property
        def T(self):
            return FakeTensor(self.data.T)
        def to(self, **kw):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.data

    class _FakeTorchModule:
        float16 = "float16"
        float32 = "float32"
        class device:
            def __init__(self, name): pass
        @staticmethod
        def from_numpy(arr):
            return FakeTensor(arr)
        @staticmethod
        def topk(tensor, k, dim=1):
            data = tensor.data
            if data.ndim == 1:
                idx = np.argsort(data)[-k:][::-1]
                return FakeTensor(data[idx]), FakeTensor(idx)
            indices = np.argsort(data, axis=dim)[:, -k:][:, ::-1].copy()
            scores = np.take_along_axis(data, indices, axis=dim).copy()
            return FakeTensor(scores), FakeTensor(indices)

    sys.modules["torch"] = _FakeTorchModule()

if EXECUTION_MODE == "api":
    # Mock adulib for api mode (no real API keys in test)
    import types

    _adulib = types.ModuleType("adulib")
    _adulib_llm = types.ModuleType("adulib.llm")
    _adulib_async = types.ModuleType("adulib.asynchronous")

    async def _fake_async_single(model, system, prompt, max_tokens=60, **kw):
        return '{"drop": [2]}', False, {}
    _adulib_llm.async_single = _fake_async_single

    async def _fake_batch_executor(fn, items, max_concurrent=10, **kw):
        results = []
        for item in items:
            results.append(await fn(item))
        return results
    _adulib_async.batch_executor = _fake_batch_executor

    sys.modules["adulib"] = _adulib
    sys.modules["adulib.llm"] = _adulib_llm
    sys.modules["adulib.asynchronous"] = _adulib_async

# ---------------------------------------------------------------------------
# 1. Synthetic upstream data
# ---------------------------------------------------------------------------
N_ONET = 20
N_JOBS = 50
DIM = 64

rng = np.random.default_rng(42)

descriptions = {
    "soc_codes": [f"11-{i:04d}" for i in range(N_ONET)],
    "titles": [f"Occupation {i}" for i in range(N_ONET)],
    "role_descriptions": [f"Role description for occupation {i}" for i in range(N_ONET)],
    "task_descriptions": [f"Task description for occupation {i}" for i in range(N_ONET)],
}

job_ads = {
    "job_ids": [f"JOB{i:06d}" for i in range(N_JOBS)],
    "role_text": [f"Job role text {i}" for i in range(N_JOBS)],
    "taskskill_text": [f"Tasks and skills for job {i}" for i in range(N_JOBS)],
    "short_desc": [f"Short desc {i}" for i in range(N_JOBS)],
    "tasks_and_skills": [f"Full tasks and skills text for job {i}" for i in range(N_JOBS)],
    "domains": [f"Sector {i % 5}" for i in range(N_JOBS)],
}

# ---------------------------------------------------------------------------
# 2. Mock ctx
# ---------------------------------------------------------------------------
ctx = SimpleNamespace(
    vars={
        "execution_mode": EXECUTION_MODE,
        "embedding_model": "mock-model",
        "embedding_dtype": "float32",
        "embed_onet_batch_size": 16,
        "embed_job_ads_batch_size": 16,
        "embed_job_ads_chunk_size": 100,
        "cosine_top_k": 5,
        "cosine_batch_size": 32,
        "llm_model": "mock-llm",
        "llm_dtype": "float32",
        "llm_batch_size": 16,
        "llm_max_new_tokens": 60,
        "llm_max_keep": 5,
        "job_ads_sample_rate": 0.5,
        "job_ads_random_seed": "42",
    }
)

# ---------------------------------------------------------------------------
# 3. Mock model loading
# ---------------------------------------------------------------------------
import isambard_utils.models as models_mod

class FakeEmbeddingModel:
    def encode(self, texts, batch_size=64, **kwargs):
        return rng.standard_normal((len(texts), DIM)).astype(np.float32)

class FakeLLM:
    def generate(self, prompts, max_new_tokens=60, **kwargs):
        return ['{"drop": [2]}'] * len(prompts)

_orig_load_emb = models_mod.load_embedding_model
_orig_load_llm = models_mod.load_llm
models_mod.load_embedding_model = lambda *a, **kw: FakeEmbeddingModel()
models_mod.load_llm = lambda *a, **kw: FakeLLM()

# ---------------------------------------------------------------------------
# 4. Run each node
# ---------------------------------------------------------------------------
try:
    print("=" * 60)
    print(f"MATCHING PIPELINE SYNTHETIC TEST (mode={EXECUTION_MODE})")
    print("=" * 60)

    # -- embed_onet --
    from ai_index.nodes.embed_onet import embed_onet
    result = embed_onet(descriptions, ctx, print)
    onet_emb = result["onet_embeddings"]
    print(f"\n>>> embed_onet OK: role={onet_emb['role_embeddings'].shape}, task={onet_emb['task_embeddings'].shape}, titles={len(onet_emb['titles'])}")

    # -- embed_job_ads --
    from ai_index.nodes.embed_job_ads import embed_job_ads
    result = embed_job_ads(job_ads, ctx, print)
    job_emb = result["job_ad_embeddings"]
    n_sampled = len(job_emb["job_ids"])
    print(f"\n>>> embed_job_ads OK: role={job_emb['role_embeddings'].shape}, task={job_emb['task_embeddings'].shape}, ids={n_sampled} (sampled from {N_JOBS})")

    # -- compute_cosine_similarity --
    from ai_index.nodes.compute_cosine_similarity import compute_cosine_similarity
    result = compute_cosine_similarity(descriptions, onet_emb, job_emb, ctx, print)
    candidates = result["candidates"]
    print(f"\n>>> compute_cosine_similarity OK: {len(candidates['job_ids'])} jobs")
    print(f"    sample candidates[0] (top 3): {candidates['candidates'][0][:3]}")

    # -- llm_filter_candidates --
    from ai_index.nodes.llm_filter_candidates import llm_filter_candidates
    result = llm_filter_candidates(candidates, job_ads, ctx, print)
    weighted = result["weighted_codes"]
    print(f"\n>>> llm_filter_candidates OK: {len(weighted['job_ids'])} jobs")
    print(f"    sample weights[0] = {weighted['weights'][0]}")

    # Verify weights sum to ~1.0
    for i, w in enumerate(weighted["weights"]):
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-6, f"Job {i}: weights sum to {total}, expected 1.0"
    print(f"    All weight vectors sum to 1.0")

    print(f"\n{'=' * 60}")
    print(f"ALL 4 MATCHING PIPELINE NODES PASSED (mode={EXECUTION_MODE})")
    print(f"  embed_onet:                {N_ONET} occupations -> ({N_ONET}, {DIM}) embeddings")
    print(f"  embed_job_ads:             {N_JOBS} -> {n_sampled} sampled -> ({n_sampled}, {DIM}) embeddings")
    print(f"  compute_cosine_similarity: {n_sampled} jobs x {N_ONET} occupations -> candidates")
    print(f"  llm_filter_candidates:     {n_sampled} jobs -> weighted O*NET codes (sum=1.0)")
    print(f"{'=' * 60}")

finally:
    models_mod.load_embedding_model = _orig_load_emb
    models_mod.load_llm = _orig_load_llm
    if "torch" in sys.modules and not hasattr(sys.modules["torch"], "__file__"):
        del sys.modules["torch"]
    for mod_name in ["adulib", "adulib.llm", "adulib.asynchronous"]:
        sys.modules.pop(mod_name, None)
