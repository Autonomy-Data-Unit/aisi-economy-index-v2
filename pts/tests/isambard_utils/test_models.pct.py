# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Integration Test: HuggingFace Model Helpers
#
# Tests for `isambard_utils.models` — model download orchestration (SSH from
# local) and model loading/inference (SBATCH on compute nodes).
#
# Uses small models for speed:
# - Embedding: `BAAI/bge-small-en-v1.5` (133MB, 384-dim)
# - LLM: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (2.2GB)
#
# **Run manually** — requires active Clifton cert and Isambard access:
# ```bash
# python -m isambard_utils_tests.test_models
# ```

# %%
#|default_exp isambard_utils.test_models

# %%
#|export
import json
import time

from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import run as ssh_run
from isambard_utils.transfer import upload_bytes
from isambard_utils.slurm import submit, wait, job_log
from isambard_utils.sbatch import generate, SbatchConfig
from isambard_utils.models import ensure_model, check_model

# %% [markdown]
# ## Helpers

# %%
#|export
_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def _print_step(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def _print_ok(msg: str = "OK"):
    print(f"  -> {msg}")

def _quote(s: str) -> str:
    import shlex
    return shlex.quote(s)

# %% [markdown]
# ## 1. check_model (not cached)

# %%
#|export
def test_check_model_not_cached(cfg: IsambardConfig):
    """Verify check_model returns False for a model not yet downloaded."""
    _print_step("1. Testing check_model (not cached)")

    # Clear any existing cache for this model to ensure a clean test
    model_dir = _EMBED_MODEL.replace("/", "--")
    cache_path = f"{cfg.hf_cache_dir}/models--{model_dir}"
    ssh_run(f"rm -rf {_quote(cache_path)}", config=cfg, check=False)

    result = check_model(_EMBED_MODEL, config=cfg)
    assert result is False, f"Expected False, got {result}"
    _print_ok(f"check_model({_EMBED_MODEL!r}) = False (correct)")

# %% [markdown]
# ## 2. ensure_model (embedding)

# %%
#|export
def test_ensure_embedding_model(cfg: IsambardConfig):
    """Download the small embedding model via ensure_model."""
    _print_step("2. Testing ensure_model (embedding)")

    path = ensure_model(_EMBED_MODEL, config=cfg)
    _print_ok(f"Snapshot path: {path}")
    assert path and len(path) > 0, "ensure_model returned empty path"
    assert "bge-small" in path.lower() or "bge" in path.lower(), \
        f"Unexpected path: {path}"
    _print_ok("Embedding model downloaded successfully")

# %% [markdown]
# ## 3. check_model (cached)

# %%
#|export
def test_check_model_cached(cfg: IsambardConfig):
    """Verify check_model returns True after download."""
    _print_step("3. Testing check_model (cached)")

    result = check_model(_EMBED_MODEL, config=cfg)
    assert result is True, f"Expected True, got {result}"
    _print_ok(f"check_model({_EMBED_MODEL!r}) = True (correct)")

# %% [markdown]
# ## 4. ensure_model (LLM)

# %%
#|export
def test_ensure_llm_model(cfg: IsambardConfig):
    """Download the small LLM via ensure_model."""
    _print_step("4. Testing ensure_model (LLM)")

    path = ensure_model(_LLM_MODEL, config=cfg)
    _print_ok(f"Snapshot path: {path}")
    assert path and len(path) > 0, "ensure_model returned empty path"
    _print_ok("LLM model downloaded successfully")

# %% [markdown]
# ## 5. Embedding inference (SBATCH)

# %%
#|export
_EMBED_TEST_SCRIPT = '''\
import json
import time

from isambard_utils.models import load_embedding_model

model_name = "BAAI/bge-small-en-v1.5"
print(f"Loading embedding model: {model_name}")

t0 = time.time()
em = load_embedding_model(model_name)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

texts = ["The cat sat on the mat.", "Machine learning is transforming industries."]
t0 = time.time()
embeddings = em.encode(texts)
encode_time = time.time() - t0

info = {
    "model": model_name,
    "device": em.device,
    "dtype": em.dtype,
    "shape": list(embeddings.shape),
    "load_time_s": round(load_time, 2),
    "encode_time_s": round(encode_time, 2),
    "embed_test_passed": embeddings.shape[0] == 2 and embeddings.shape[1] == 384,
}

print(f"Embeddings shape: {embeddings.shape}")
print("EMBED_TEST_RESULT:" + json.dumps(info))
'''

# %%
#|export
def test_embedding_inference(cfg: IsambardConfig):
    """Run embedding model inference on GPU via SBATCH."""
    _print_step("5. Testing embedding inference on GPU")

    remote_script = f"{cfg.project_dir}/.test_embed_models.py"
    upload_bytes(_EMBED_TEST_SCRIPT.encode(), remote_script, config=cfg)
    _print_ok("Uploaded embedding test script")

    sc = SbatchConfig(
        job_name="iutils_embed_model_test",
        time="00:10:00",
        python_script=remote_script,
        env_vars={"TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"},
    )
    script = generate(sc, isambard_config=cfg)
    job = submit(script, config=cfg)
    _print_ok(f"Submitted job: {job.job_id}")

    _print_ok("Waiting for job to complete...")
    def _on_poll(s):
        if s:
            print(f"    [{time.strftime('%H:%M:%S')}] state={s.get('state', '?')}, node={s.get('node', '?')}")
        else:
            print(f"    [{time.strftime('%H:%M:%S')}] job left queue, checking sacct...")

    result = wait(job.job_id, config=cfg, poll_interval=10, timeout=300, on_poll=_on_poll)
    _print_ok(f"Job finished: state={result.get('state', '?')}, exit_code={result.get('exit_code', '?')}")

    stdout = job_log(job.job_id, config=cfg, stream="stdout")
    stderr = job_log(job.job_id, config=cfg, stream="stderr")

    if stderr.strip():
        stderr_lines = stderr.strip().split("\n")
        if len(stderr_lines) > 10:
            print(f"  stderr (last 10 of {len(stderr_lines)} lines):")
            for line in stderr_lines[-10:]:
                print(f"    {line}")
        else:
            print(f"  stderr:\n{stderr}")

    _print_ok(f"stdout ({len(stdout)} chars):")
    for line in stdout.strip().split("\n"):
        print(f"    {line}")

    for line in stdout.strip().split("\n"):
        if line.startswith("EMBED_TEST_RESULT:"):
            embed_info = json.loads(line.split("EMBED_TEST_RESULT:", 1)[1])
            _print_ok(f"Embedding info: {json.dumps(embed_info, indent=2)}")
            assert embed_info["embed_test_passed"], "Embedding test failed"
            assert embed_info["shape"] == [2, 384], f"Unexpected shape: {embed_info['shape']}"
            _print_ok(f"Embedding verified: shape={embed_info['shape']}, {embed_info['encode_time_s']}s")
            break
    else:
        raise AssertionError(f"EMBED_TEST_RESULT not found in job stdout:\n{stdout}")

    ssh_run(f"rm -f {remote_script}", config=cfg)
    _print_ok("Cleaned up test script")

# %% [markdown]
# ## 6. LLM inference (SBATCH)

# %%
#|export
_LLM_TEST_SCRIPT = '''\
import json
import time

from isambard_utils.models import load_llm

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading LLM: {model_name}")

t0 = time.time()
llm = load_llm(model_name)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

t0 = time.time()
responses = llm.generate("What is 2+2? Answer in one short sentence.", max_new_tokens=50)
gen_time = time.time() - t0

response = responses[0]
import torch
info = {
    "model": model_name,
    "device": llm.device,
    "dtype": llm.dtype,
    "gpu_mem_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
    "load_time_s": round(load_time, 2),
    "gen_time_s": round(gen_time, 2),
    "response": response,
    "response_len": len(response),
    "llm_test_passed": len(response) > 0,
}

print(f"Response: {response}")
print("LLM_TEST_RESULT:" + json.dumps(info))
'''

# %%
#|export
def test_llm_inference(cfg: IsambardConfig):
    """Run LLM inference on GPU via SBATCH."""
    _print_step("6. Testing LLM inference on GPU (TinyLlama 1.1B)")

    remote_script = f"{cfg.project_dir}/.test_llm_models.py"
    upload_bytes(_LLM_TEST_SCRIPT.encode(), remote_script, config=cfg)
    _print_ok("Uploaded LLM test script")

    sc = SbatchConfig(
        job_name="iutils_llm_model_test",
        time="00:15:00",
        mem="80G",
        python_script=remote_script,
        env_vars={"TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"},
    )
    script = generate(sc, isambard_config=cfg)
    job = submit(script, config=cfg)
    _print_ok(f"Submitted job: {job.job_id}")

    _print_ok("Waiting for job to complete...")
    def _on_poll(s):
        if s:
            print(f"    [{time.strftime('%H:%M:%S')}] state={s.get('state', '?')}, node={s.get('node', '?')}")
        else:
            print(f"    [{time.strftime('%H:%M:%S')}] job left queue, checking sacct...")

    result = wait(job.job_id, config=cfg, poll_interval=10, timeout=600, on_poll=_on_poll)
    _print_ok(f"Job finished: state={result.get('state', '?')}, exit_code={result.get('exit_code', '?')}")

    stdout = job_log(job.job_id, config=cfg, stream="stdout")
    stderr = job_log(job.job_id, config=cfg, stream="stderr")

    if stderr.strip():
        stderr_lines = stderr.strip().split("\n")
        if len(stderr_lines) > 10:
            print(f"  stderr (last 10 of {len(stderr_lines)} lines):")
            for line in stderr_lines[-10:]:
                print(f"    {line}")
        else:
            print(f"  stderr:\n{stderr}")

    _print_ok(f"stdout ({len(stdout)} chars):")
    for line in stdout.strip().split("\n"):
        print(f"    {line}")

    for line in stdout.strip().split("\n"):
        if line.startswith("LLM_TEST_RESULT:"):
            llm_info = json.loads(line.split("LLM_TEST_RESULT:", 1)[1])
            _print_ok(f"LLM info: {json.dumps(llm_info, indent=2)}")
            assert llm_info["llm_test_passed"], "LLM test failed"
            assert llm_info["response_len"] > 0, "No response generated"
            _print_ok(
                f"LLM verified: {llm_info['model']}, "
                f"{llm_info['gen_time_s']}s, "
                f"{llm_info['gpu_mem_allocated_mb']} MB GPU mem"
            )
            break
    else:
        raise AssertionError(f"LLM_TEST_RESULT not found in job stdout:\n{stdout}")

    ssh_run(f"rm -f {remote_script}", config=cfg)
    _print_ok("Cleaned up test script")

# %% [markdown]
# ## Run All Tests

# %%
#|export
def run_all():
    """Run the full model helpers integration test suite."""
    print("Isambard Utils — Model Helpers Integration Tests")
    print(f"{'='*60}")

    cfg = IsambardConfig.from_env()
    print(f"Host:        {cfg.ssh_host}")
    print(f"Project dir: {cfg.project_dir}")
    print(f"HF cache:    {cfg.hf_cache_dir}")

    tests = [
        test_check_model_not_cached,
        test_ensure_embedding_model,
        test_check_model_cached,
        test_ensure_llm_model,
        test_embedding_inference,
        test_llm_inference,
    ]

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn(cfg)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()

    _print_step(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)

# %%
#|export
if __name__ == "__main__":
    run_all()
