# /projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store/isambard/202601/run_llm_extract_month2.py
# -*- coding: utf-8 -*-

import os
import re
import json
import gc
import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# CONFIG (Isambard)
# =============================================================================
PROJECT = Path("/projects/a5u/adu_dev/aisi-economy-index")

# required context from sbatch
MONTH = os.environ["MONTH"]
NPZ_PATH = Path(os.environ["NPZ_PATH"])

# slice boundaries
START = int(os.environ.get("START", 0))
STOP  = int(os.environ.get("STOP", 50000))

# output lives next to input month file
OUT_DIR = NPZ_PATH.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_JSONL = OUT_DIR / f"{MONTH}_extract_{START}_{STOP}_{ts}.jsonl"

# model + generation params
HF_MODEL_DIR = PROJECT / "hf_cache" / "models--meta-llama--Meta-Llama-3.1-8B-Instruct"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", 220))

# speed knobs (no prompt change)
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", 16))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "1") == "1"
COMPILE = os.environ.get("TORCH_COMPILE", "0") == "1"
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "1") == "1"

# cap input token length at tokenization time
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", 2048))

# =============================================================================
# SCHEMA (aligned with Colab)
# =============================================================================
class JobInfoModel(BaseModel):
    short_description: str
    tasks: List[str]
    skills: List[str]
    domain: str
    level: str
    automation_prof_score: int


# =============================================================================
# JSON EXTRACT (robust)
# =============================================================================
_JSON_RE = re.compile(r"\{[\s\S]*\}", flags=re.DOTALL)

def extract_json_object(text: str) -> str:
    cleaned = re.sub(r"```json|```", "", text).strip()
    m = _JSON_RE.search(cleaned)
    if not m:
        raise ValueError("No JSON found")
    return m.group(0).strip()


# =============================================================================
# MODEL PATH RESOLUTION
# =============================================================================
def resolve_snapshot_dir(model_root: Path) -> Path:
    snap_root = model_root / "snapshots"
    if not snap_root.exists():
        raise FileNotFoundError(f"Missing snapshots dir: {snap_root}")
    snaps = [p for p in snap_root.iterdir() if p.is_dir()]
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under: {snap_root}")
    snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0]


# =============================================================================
# PROMPT (unchanged)
# =============================================================================
SYSTEM_PROMPT = (
    "You are a human resources highly-accurate data extraction bot. "
    "Extract the following details from the job advertisement provided by the user. "
    "You MUST NOT include more than 5 tasks or 5 skills. Stop the list at 5. Do not write more. "
    "Your response MUST be a single, valid JSON object (no markdown, no explanation, no code block). "
    "The JSON must have exactly these keys: "
    "'short_description', 'tasks', 'skills', 'domain', 'level', 'automation_prof_score'. "
    "- 'level': classify as 'Entry-Level' if the job requires <3 years experience or mentions 'junior'/'entry'; otherwise 'Experienced'. "
    "- 'automation_prof_score': integer 0–10 estimating AI automation risk. "
    "0 = AI-proof (requires physical/manual presence, creativity, leadership, or deep social judgment). "
    "10 = highly automatable by AI (routine, repetitive non-manual tasks like data entry, scheduling). "
    "Manual labour (e.g. cleaning, lifting, warehouse, driving) should usually score 0–3, "
    "since AI alone cannot replace them."
)

USER_TEMPLATE = """Extract:
1. Short job description
2. Bullet list of up to 5 key tasks
3. Bullet list of up to 5 key skills
4. The domain or industry
5. The level (Entry-Level if <3 years experience or junior, else Experienced)
6. Automation proof score (0=AI proof, 10=highly automatable non-manual tasks)

Job Ad:
{job_text}
"""

def build_prompt(tokenizer, title: str, category: str, description: str) -> str:
    job_text = f"{title}\n{category}\n\n{(description or '')[:1200]}"  # unchanged
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(job_text=job_text)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# =============================================================================
# MODEL LOADER (speed: SDPA + optional compile)
# =============================================================================
def load_model():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

    snapshot = resolve_snapshot_dir(HF_MODEL_DIR)
    print("[MODEL] snapshot:", snapshot)

    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    attn_impl = "sdpa" if FLASH_ATTN else None

    mdl = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    ).to("cuda")
    mdl.eval()

    if COMPILE:
        try:
            mdl = torch.compile(mdl)
            print("[MODEL] torch.compile enabled")
        except Exception as e:
            print("[MODEL] torch.compile failed, continuing:", repr(e))

    return mdl, tok


# =============================================================================
# MAIN
# =============================================================================
def main():
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("[ENV] torch:", torch.__version__)
    print("[ENV] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[ENV] gpu:", torch.cuda.get_device_name(0))
    print(f"[ENV] START={START} STOP={STOP} BATCH_SIZE={BATCH_SIZE} GEN_MAX_TOKENS={GEN_MAX_TOKENS}")
    print(f"[ENV] TORCH_THREADS={TORCH_THREADS} FLASH_ATTN={FLASH_ATTN} TORCH_COMPILE={COMPILE}")
    print(f"[ENV] MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}")

    print("[LOAD] npz:", NPZ_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)

    ids_all = data["id"]
    cats_all = data["category_name"]
    titles_all = data["title"]
    descs_all = data["description"]

    n = len(ids_all)
    a = max(0, START)
    b = min(STOP, n)
    print(f"[RANGE] {a}:{b} (n={n}) -> {b-a} rows")
    print("[OUT] jsonl:", OUT_JSONL)

    model, tokenizer = load_model()

    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

    with open(OUT_JSONL, "w", encoding="utf-8", buffering=1) as fout:
        for b0 in tqdm(range(a, b, BATCH_SIZE), desc="Batches"):
            b1 = min(b0 + BATCH_SIZE, b)

            prompts = []
            meta = []

            for i in range(b0, b1):
                jid = str(ids_all[i])
                cat = str(cats_all[i])
                ttl = str(titles_all[i])
                desc = str(descs_all[i])

                prompts.append(build_prompt(tokenizer, ttl, cat, desc))
                meta.append((jid, cat, ttl, desc[:1200]))

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            ).to(model.device)

            ctx = torch.inference_mode if INFERENCE_MODE else torch.no_grad
            with ctx():
                out = model.generate(
                    **inputs,
                    max_new_tokens=GEN_MAX_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            replies = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            for (jid, cat, ttl, job_desc), reply in zip(meta, replies):
                parsed = None
                error = None
                raw = reply

                try:
                    obj = extract_json_object(raw)
                    data_obj = json.loads(obj)

                    data_obj["tasks"] = (data_obj.get("tasks") or [])[:5]
                    data_obj["skills"] = (data_obj.get("skills") or [])[:5]

                    validated = JobInfoModel.model_validate(data_obj)
                    parsed = validated.model_dump()
                except (json.JSONDecodeError, ValidationError, Exception) as e:
                    error = f"{type(e).__name__}: {e}"

                fout.write(json.dumps({
                    "id": jid,
                    "category_name": cat,
                    "title": ttl,
                    "job_description": job_desc,
                    "llm_output": raw,
                    "parsed": parsed,
                    "error": error
                }, ensure_ascii=False) + "\n")

            fout.flush()
            if (b0 // BATCH_SIZE) % 10 == 0:
                try:
                    os.fsync(fout.fileno())
                except Exception:
                    pass

            del inputs, out, replies, prompts, meta

            if (b0 // BATCH_SIZE) % 25 == 0:
                torch.cuda.empty_cache()

    print("[DONE] wrote:", OUT_JSONL)
    if torch.cuda.is_available():
        print("Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
        print("Reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")


if __name__ == "__main__":
    main()
