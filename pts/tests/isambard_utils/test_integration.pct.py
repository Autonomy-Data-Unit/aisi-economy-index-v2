# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Integration Test: Isambard Utils
#
# End-to-end test that verifies the full workflow: SSH into Isambard, set up
# the environment, submit a GPU job via Slurm, wait for completion, and verify
# output. Requires active Clifton certificate auth and network access.
#
# **Run manually** — this is not part of the CI test suite since it needs a live
# Isambard connection. Execute the notebook or run:
# ```bash
# python -m isambard_utils_tests.test_integration
# ```

# %%
#|default_exp isambard_utils.test_integration

# %%
#|export
import os
import time
import tempfile
from pathlib import Path

from isambard_utils.config import IsambardConfig
from isambard_utils.ssh import run as ssh_run, check_connection
from isambard_utils.transfer import upload, download, upload_bytes
from isambard_utils.slurm import submit, status, wait, cancel, job_log, SlurmJob
from isambard_utils.sbatch import generate, SbatchConfig
from isambard_utils.orchestrate import setup_runner

# %% [markdown]
# ## Helpers

# %%
#|export
def _print_step(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")

def _print_ok(msg: str = "OK"):
    print(f"  -> {msg}")

def _quote(s: str) -> str:
    import shlex
    return shlex.quote(s)

# %% [markdown]
# ## 1. SSH Connectivity

# %%
#|export
def test_ssh_connection(cfg: IsambardConfig):
    """Verify we can SSH into the Isambard login node."""
    _print_step("1. Testing SSH connection")

    assert check_connection(cfg), "SSH connection failed — is your Clifton cert active?"
    _print_ok("SSH connection established")

    result = ssh_run("hostname", config=cfg)
    hostname = result.stdout.strip()
    _print_ok(f"Login node hostname: {hostname}")

    result = ssh_run("uname -m", config=cfg)
    arch = result.stdout.strip()
    _print_ok(f"Architecture: {arch}")
    assert arch == "aarch64", f"Expected aarch64, got {arch}"

# %% [markdown]
# ## 2. File Transfer

# %%
#|export
def test_file_transfer(cfg: IsambardConfig):
    """Verify upload and download of files via rsync and SSH pipe."""
    _print_step("2. Testing file transfer")

    test_dir = f"{cfg.project_dir}/.test_transfer"
    ssh_run(f"mkdir -p {test_dir}", config=cfg)

    # Test upload_bytes (small data via SSH pipe)
    test_content = b"hello from isambard_utils integration test\n"
    remote_file = f"{test_dir}/test_upload.txt"
    upload_bytes(test_content, remote_file, config=cfg)
    _print_ok("upload_bytes: wrote test file")

    result = ssh_run(f"cat {remote_file}", config=cfg)
    assert result.stdout.strip() == test_content.decode().strip(), \
        f"Content mismatch: {result.stdout!r}"
    _print_ok("upload_bytes: content verified on remote")

    # Test download via rsync
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = os.path.join(tmpdir, "test_upload.txt")
        download(remote_file, local_file, config=cfg)
        downloaded = Path(local_file).read_bytes()
        assert downloaded == test_content, f"Download mismatch: {downloaded!r}"
        _print_ok("download: content matches original")

    # Test upload via rsync (directory)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small test directory locally
        test_subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(test_subdir)
        Path(os.path.join(test_subdir, "a.txt")).write_text("file a\n")
        Path(os.path.join(test_subdir, "b.txt")).write_text("file b\n")

        remote_subdir = f"{test_dir}/synced_dir"
        upload(test_subdir + "/", remote_subdir + "/", config=cfg)

        result = ssh_run(f"ls {remote_subdir}", config=cfg)
        files = result.stdout.strip().split("\n")
        assert "a.txt" in files and "b.txt" in files, f"Expected a.txt, b.txt; got {files}"
        _print_ok(f"upload (rsync dir): synced {len(files)} files")

    # Cleanup
    ssh_run(f"rm -rf {test_dir}", config=cfg)
    _print_ok("Cleaned up test files")

# %% [markdown]
# ## 3. Environment Setup

# %%
#|export
def test_environment_setup(cfg: IsambardConfig):
    """Verify environment bootstrap via setup_runner (syncs llm_runner, ensures venv + CUDA torch)."""
    _print_step("3. Testing environment setup")

    setup_runner(config=cfg)
    _print_ok("setup_runner() completed")

    # Verify directories
    ssh_run(f"test -d {cfg.logs_dir}", config=cfg)
    _print_ok(f"logs_dir exists: {cfg.logs_dir}")
    ssh_run(f"test -d {cfg.hf_cache_dir}", config=cfg)
    _print_ok(f"hf_cache_dir exists: {cfg.hf_cache_dir}")

    # Verify venv exists
    ssh_run(f"test -f {cfg.project_dir}/.venv/bin/python", config=cfg)
    _print_ok("venv exists")

# %% [markdown]
# ## 4. SBATCH Script Generation

# %%
#|export
def test_sbatch_generation(cfg: IsambardConfig):
    """Verify SBATCH script generation produces valid content."""
    _print_step("4. Testing SBATCH script generation")

    sc = SbatchConfig(
        job_name="iutils_test",
        time="00:05:00",
        python_command='import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"Device: {torch.cuda.get_device_name(0)}")',
    )
    script = generate(sc, isambard_config=cfg)
    _print_ok("Generated SBATCH script:")
    for line in script.strip().split("\n"):
        print(f"    {line}")

    # Basic sanity checks on generated script
    assert "#!/bin/bash" in script
    assert "#SBATCH --job-name=iutils_test" in script
    assert "#SBATCH --time=00:05:00" in script
    assert "#SBATCH --gpus=1" in script
    assert "module load cudatoolkit/24.11_12.6" in script
    assert "source .venv/bin/activate" in script
    assert "srun python -c" in script
    assert "torch.cuda" in script
    _print_ok("Script content validated")

# %% [markdown]
# ## 5. GPU Job Submission + Execution

# %%
#|export
_GPU_TEST_SCRIPT = '''\
import torch
import sys
import json

info = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}

if torch.cuda.is_available():
    info["device_name"] = torch.cuda.get_device_name(0)
    info["cuda_version"] = torch.version.cuda

    # Quick tensor operation on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y
    info["matmul_shape"] = list(z.shape)
    info["matmul_sum"] = float(z.sum().cpu())
    info["gpu_test_passed"] = True
else:
    info["gpu_test_passed"] = False
    print("WARNING: CUDA not available!", file=sys.stderr)

print("GPU_TEST_RESULT:" + json.dumps(info))
'''

# %%
#|export
def test_gpu_job(cfg: IsambardConfig):
    """Submit a GPU job, wait for completion, and verify GPU access."""
    _print_step("5. Testing GPU job submission + execution")

    # Upload the test script
    remote_script = f"{cfg.project_dir}/.test_gpu_check.py"
    upload_bytes(_GPU_TEST_SCRIPT.encode(), remote_script, config=cfg)
    _print_ok("Uploaded GPU test script")

    # Generate and submit SBATCH job
    sc = SbatchConfig(
        job_name="iutils_gpu_test",
        time="00:10:00",
        python_script=remote_script,
    )
    script = generate(sc, isambard_config=cfg)
    _print_ok("Generated SBATCH script")

    job = submit(script, config=cfg)
    _print_ok(f"Submitted job: {job.job_id}")

    # Check initial status
    job_status = status(job.job_id, config=cfg)
    _print_ok(f"Initial status: {job_status.get('state', 'queued/unknown')}")

    # Wait for completion
    _print_ok("Waiting for job to complete (polling every 10s)...")
    def _on_poll(s):
        if s:
            print(f"    [{time.strftime('%H:%M:%S')}] state={s.get('state', '?')}, node={s.get('node', '?')}")
        else:
            print(f"    [{time.strftime('%H:%M:%S')}] job left queue, checking sacct...")

    result = wait(job.job_id, config=cfg, poll_interval=10, timeout=300, on_poll=_on_poll)
    _print_ok(f"Job finished: state={result.get('state', '?')}, exit_code={result.get('exit_code', '?')}")

    # Read and verify output
    stdout = job_log(job.job_id, config=cfg, stream="stdout")
    stderr = job_log(job.job_id, config=cfg, stream="stderr")

    if stderr.strip():
        print(f"  stderr:\n{stderr}")

    _print_ok(f"stdout ({len(stdout)} chars):")
    for line in stdout.strip().split("\n"):
        print(f"    {line}")

    # Parse the GPU test result
    import json
    for line in stdout.strip().split("\n"):
        if line.startswith("GPU_TEST_RESULT:"):
            gpu_info = json.loads(line.split("GPU_TEST_RESULT:", 1)[1])
            _print_ok(f"GPU info: {json.dumps(gpu_info, indent=2)}")
            assert gpu_info["gpu_test_passed"], "GPU test failed — CUDA not available on compute node"
            assert gpu_info["cuda_available"], "torch.cuda.is_available() returned False"
            assert gpu_info["device_count"] >= 1, "No CUDA devices found"
            _print_ok(f"GPU verified: {gpu_info.get('device_name', 'unknown')}, CUDA {gpu_info.get('cuda_version', '?')}")
            break
    else:
        raise AssertionError(f"GPU_TEST_RESULT not found in job stdout:\n{stdout}")

    # Cleanup
    ssh_run(f"rm -f {remote_script}", config=cfg)
    _print_ok("Cleaned up test script")

# %% [markdown]
# ## 6. Job Cancellation

# %%
#|export
def test_job_cancel(cfg: IsambardConfig):
    """Submit a long-running job and cancel it to verify scancel works."""
    _print_step("6. Testing job cancellation")

    sc = SbatchConfig(
        job_name="iutils_cancel_test",
        time="00:10:00",
        python_command="import time; time.sleep(600)",
    )
    script = generate(sc, isambard_config=cfg)
    job = submit(script, config=cfg)
    _print_ok(f"Submitted long-running job: {job.job_id}")

    # Give it a moment to enter the queue
    time.sleep(3)

    job_status = status(job.job_id, config=cfg)
    state = job_status.get("state", "")
    _print_ok(f"Job state before cancel: {state}")

    # Cancel it
    cancel(job.job_id, config=cfg)
    _print_ok("scancel sent")

    # Wait briefly and verify it's gone/cancelled
    time.sleep(5)
    job_status = status(job.job_id, config=cfg)
    if job_status:
        state = job_status.get("state", "")
        assert state in ("CANCELLED", "COMPLETING", ""), \
            f"Expected CANCELLED, got {state}"
    _print_ok("Job cancelled successfully")

# %% [markdown]
# ## 7. LLM Inference on GPU
#
# Run a small language model (TinyLlama 1.1B) on the GH200 GPU to verify
# that the full HuggingFace + transformers + CUDA stack works end-to-end.

# %%
#|export
_LLM_TEST_SCRIPT = '''\
import torch
import json
import sys
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading model: {model_name}")

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

# Format as chat
prompt = "<|user|>\\nWhat is 2+2? Answer in one short sentence.</s>\\n<|assistant|>\\n"

t0 = time.time()
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0,
    )
gen_time = time.time() - t0

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
n_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

info = {
    "model": model_name,
    "device": str(model.device),
    "dtype": str(model.dtype),
    "gpu_mem_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
    "load_time_s": round(load_time, 2),
    "gen_time_s": round(gen_time, 2),
    "tokens_generated": n_tokens,
    "tokens_per_second": round(n_tokens / gen_time, 1) if gen_time > 0 else 0,
    "response": response,
    "llm_test_passed": True,
}

print(f"Response: {response}")
print(f"Generated {n_tokens} tokens in {gen_time:.2f}s ({info['tokens_per_second']} tok/s)")
print("LLM_TEST_RESULT:" + json.dumps(info))
'''

# %%
#|export
def test_llm_inference(cfg: IsambardConfig):
    """Run a small LLM on GPU to verify the full inference stack."""
    _print_step("7. Testing LLM inference on GPU (TinyLlama 1.1B)")

    # Upload the test script
    remote_script = f"{cfg.project_dir}/.test_llm_check.py"
    upload_bytes(_LLM_TEST_SCRIPT.encode(), remote_script, config=cfg)
    _print_ok("Uploaded LLM test script")

    # Generate and submit SBATCH job
    # Request slightly more time + memory for model download on first run
    sc = SbatchConfig(
        job_name="iutils_llm_test",
        time="00:15:00",
        mem="80G",
        python_script=remote_script,
        env_vars={"TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"},
    )
    script = generate(sc, isambard_config=cfg)
    _print_ok("Generated SBATCH script")

    job = submit(script, config=cfg)
    _print_ok(f"Submitted job: {job.job_id}")

    # Wait for completion (longer timeout — model download may take a while first time)
    _print_ok("Waiting for job to complete (polling every 10s, up to 10min)...")
    def _on_poll(s):
        if s:
            print(f"    [{time.strftime('%H:%M:%S')}] state={s.get('state', '?')}, node={s.get('node', '?')}")
        else:
            print(f"    [{time.strftime('%H:%M:%S')}] job left queue, checking sacct...")

    result = wait(job.job_id, config=cfg, poll_interval=10, timeout=600, on_poll=_on_poll)
    _print_ok(f"Job finished: state={result.get('state', '?')}, exit_code={result.get('exit_code', '?')}")

    # Read output
    stdout = job_log(job.job_id, config=cfg, stream="stdout")
    stderr = job_log(job.job_id, config=cfg, stream="stderr")

    if stderr.strip():
        # Transformers prints a lot of warnings to stderr, show tail only
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

    # Parse the LLM test result
    import json
    for line in stdout.strip().split("\n"):
        if line.startswith("LLM_TEST_RESULT:"):
            llm_info = json.loads(line.split("LLM_TEST_RESULT:", 1)[1])
            _print_ok(f"LLM info: {json.dumps(llm_info, indent=2)}")
            assert llm_info["llm_test_passed"], "LLM test failed"
            assert llm_info["tokens_generated"] > 0, "No tokens generated"
            _print_ok(
                f"LLM verified: {llm_info['model']} on {llm_info['device']}, "
                f"{llm_info['tokens_per_second']} tok/s, "
                f"{llm_info['gpu_mem_allocated_mb']} MB GPU mem"
            )
            break
    else:
        raise AssertionError(f"LLM_TEST_RESULT not found in job stdout:\n{stdout}")

    # Cleanup
    ssh_run(f"rm -f {remote_script}", config=cfg)
    _print_ok("Cleaned up test script")

# %% [markdown]
# ## Run All Tests

# %%
#|export
def run_all():
    """Run the full integration test suite."""
    print("Isambard Utils — Integration Test Suite")
    print(f"{'='*60}")

    cfg = IsambardConfig.from_env()
    print(f"Host:        {cfg.ssh_host}")
    print(f"Project dir: {cfg.project_dir}")

    tests = [
        test_ssh_connection,
        test_file_transfer,
        test_environment_setup,
        test_sbatch_generation,
        test_gpu_job,
        test_llm_inference,
        test_job_cancel,
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
#|eval: false
if __name__ == "__main__":
    run_all()
