# -*- coding: utf-8 -*-
import os
import json
import datetime
import time
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# =============================================================================
# CONFIG (Isambard)
# =============================================================================
PROJECT = Path("/projects/a5u/adu_dev/aisi-economy-index")
BASE = PROJECT / "aisi_economy_index/store/AISI_demo/stage_3/dev"

NPZ_PATH = Path(os.environ["NPZ_PATH"])
MONTH_TAG = os.environ.get("MONTH_TAG", NPZ_PATH.stem)

BRICS_CACHE = Path("/projects/public/brics/cache")
MODEL_ID = os.environ.get("MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

START = int(os.environ.get("START", 0))
STOP = int(os.environ.get("STOP", 1000))

# Test mode: cap rows (default 100)
TEST_ONLY = int(os.environ.get("TEST_ONLY", 1))  # 1 cap, 0 full
TEST_N = int(os.environ.get("TEST_N", 100))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))

# Pay the reasoning tax for R1
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", 1500))

# Retry settings: smaller, more "JSON now" oriented
RETRY_MAX_TOKENS = int(os.environ.get("RETRY_MAX_TOKENS", 300))

MAX_KEEP = int(os.environ.get("MAX_KEEP", 4))
MIN_KEEP = int(os.environ.get("MIN_KEEP", 2))

DO_SAMPLE = int(os.environ.get("DO_SAMPLE", 1))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.6))
TOP_P = float(os.environ.get("TOP_P", 0.95))

jobid = os.environ.get("SLURM_JOB_ID", "nojid")
taskid = os.environ.get("SLURM_ARRAY_TASK_ID", "notask")
EMBED = os.environ.get("EMBED", "unknown_embed")

OUT_DIR = BASE / "llm_negative_selection" / "deepseek" / EMBED / MONTH_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
rank = os.environ.get("SLURM_PROCID", "0")
OUT_JSONL = OUT_DIR / f"deepseek_drop_only_{MONTH_TAG}_{START}_{STOP}_job{jobid}_task{taskid}_rank{rank}_{ts}.jsonl"

# =============================================================================
# SCHEMA
# =============================================================================
class DropResponse(BaseModel):
    drop: list[int]

# =============================================================================
# JSON EXTRACTION (ROBUST TO UNFINISHED <think>)
# =============================================================================
def extract_last_json_object(text: str) -> str:
    """
    Extract the last balanced {...} JSON object from arbitrary model output.
    Robust to:
      - missing </think>
      - multiple JSON objects
      - pre/post garbage
    """
    if not text:
        raise ValueError("Empty reply")

    # If <think> exists and a closing tag exists, drop everything up to the closing tag.
    # If closing tag is missing, do NOT regex-strip (can delete JSON); just proceed.
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1]

    start, depth, last_obj = None, 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last_obj = text[start : i + 1]

    if last_obj is None:
        raise ValueError("No JSON object found")
    return last_obj.strip()

# =============================================================================
# MODEL LOADER (OFFLINE)
# =============================================================================
def load_model():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=str(BRICS_CACHE),
        local_files_only=True,
    )
    print("[MODEL] path:", model_path, flush=True)

    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    )
    mdl.eval()
    return mdl, tok

# =============================================================================
# PROMPTS
# =============================================================================
def build_prompt(tasks_str, clean_titles, domain, job_ad_title, job_sector_category, full_ad_text) -> str:
    n = len(clean_titles)
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(clean_titles))

    excerpt = (full_ad_text or "").strip()[:500]
    full_block = f"\nAD EXCERPT (tools/duties evidence only):\n{excerpt}\n" if excerpt else ""

    return f"""
ROLE
You are auditing occupation matches. Your job is to KEEP 2–4 candidates whenever plausible.

RANKING NOTE
Candidates are ranked by cosine similarity from an embedding model (rank 1 = highest cosine).
This ranking is a weak prior. Do NOT assume rank 1 is correct.

INPUTS (SOURCE OF TRUTH)
Job title: {job_ad_title}
Sector: {job_sector_category}
Domain: {domain}
Tasks: {tasks_str}
{full_block}

CANDIDATES (1-based, similarity-ranked)
{numbered}

DECISION RULES
1) Keep 2–4 by default. Keeping only 1 is rare.
2) First, identify the FUNCTIONAL ANCHOR from title + tasks. Keep it unless clearly contradicted.
3) Then actively find 1–3 additional plausible matches:
   - same functional family or adjacent duties
   - same occupation at different level (technician vs engineer etc)
   - tasks overlap partially
4) Drop candidates that are clearly wrong functionally, or generic cross-sector roles unless tasks justify.
5) Manager rule: keep manager roles only if tasks mention supervision, rotas, hiring, budgeting.
6) IT lock: if title/tasks mention concrete tech (Python/SQL/APIs/systems), keep relevant IT roles.

OUTPUT INSTRUCTION
1) You MUST first think inside <think> tags.
2) After thinking, output the final JSON object exactly: {{"drop":[...]}}
""".strip()

def build_retry_prompt(tasks_str, clean_titles, domain, job_ad_title, job_sector_category) -> str:
    n = len(clean_titles)
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(clean_titles))
    return f"""
Return ONLY valid JSON with key "drop": {{"drop":[...]}}.
No <think>. No prose. No markdown.

Job title: {job_ad_title}
Sector: {job_sector_category}
Domain: {domain}
Tasks: {tasks_str}

Candidates (1-based):
{numbered}

Rules:
- Keep 2–4 if plausible.
- Drop only clearly wrong functions.
""".strip()

# =============================================================================
# UTILS
# =============================================================================
def normalise_tasks(cur_tasks) -> str:
    if isinstance(cur_tasks, (list, np.ndarray)):
        xs = [str(t).strip() for t in cur_tasks if t and str(t).strip()]
        return ", ".join(xs)[:800]
    return str(cur_tasks).strip()[:800]

def gen_kwargs(do_sample: bool, temperature: float, top_p: float, max_new_tokens: int, pad_token_id: int):
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=pad_token_id)
    if do_sample:
        kw.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        kw.update(dict(do_sample=False))
    return kw

def parse_drop_indices(reply: str, n_titles: int) -> list[int]:
    jtxt = extract_last_json_object(reply)
    raw_drop = DropResponse.model_validate_json(jtxt).drop
    return [int(x) for x in raw_drop if 1 <= int(x) <= n_titles]

# =============================================================================
# MAIN
# =============================================================================
def main():
    start_wall = time.time()
    print("[START]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[LOAD] npz:", NPZ_PATH, flush=True)

    data = np.load(NPZ_PATH, allow_pickle=True)
    required = ["job_ids", "titles", "job_ad_title", "job_sector_category", "job_tasks", "domain", "job_description"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"NPZ missing keys: {missing}. Found: {sorted(data.files)}")

    job_ids = data["job_ids"]
    candidates = data["titles"]
    job_ad_titles = data["job_ad_title"]
    job_sector_categories = data["job_sector_category"]
    job_tasks = data["job_tasks"]
    domains = data["domain"]
    job_full_ads = data["job_description"]

    n_rows = len(job_ids)
    a = max(0, START)
    b = min(STOP, n_rows)
    if TEST_ONLY:
        b = min(b, a + TEST_N)
    print(f"[RANGE] {a}:{b} -> {b-a} jobs (TEST_ONLY={TEST_ONLY}, TEST_N={TEST_N})", flush=True)

    model, tokenizer = load_model()

    system_msg = "Follow user instructions carefully."
    retry_system_msg = "Return ONLY valid JSON. No extra text."

    pad_id = tokenizer.eos_token_id

    def run_one(prompt_text: str, max_new_tokens: int, do_sample_local: bool) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                **gen_kwargs(DO_SAMPLE if do_sample_local else False, TEMPERATURE, TOP_P, max_new_tokens, pad_id),
            )
        new_tokens = out[:, inputs.input_ids.shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    with open(OUT_JSONL, "w", encoding="utf-8") as fout:
        for b0 in tqdm(range(a, b, BATCH_SIZE), desc="Batches"):
            b1 = min(b0 + BATCH_SIZE, b)

            prompts, meta = [], []
            for i in range(b0, b1):
                jid = str(job_ids[i])

                titles_raw = list(candidates[i]) if candidates[i] is not None else []
                clean_titles = [str(t).strip() for t in titles_raw if t and str(t).strip()]

                if not clean_titles:
                    fout.write(json.dumps({"job_id": jid, "candidates": [], "drop": [], "final": []}, ensure_ascii=False) + "\n")
                    continue

                tasks_str = normalise_tasks(job_tasks[i])

                p = build_prompt(
                    tasks_str=tasks_str,
                    clean_titles=clean_titles,
                    domain=str(domains[i]),
                    job_ad_title=str(job_ad_titles[i]),
                    job_sector_category=str(job_sector_categories[i]),
                    full_ad_text=str(job_full_ads[i]) if job_full_ads[i] is not None else "",
                )
                prompts.append(p)
                meta.append((jid, clean_titles, tasks_str, str(domains[i]), str(job_ad_titles[i]), str(job_sector_categories[i])))

            if not prompts:
                continue

            # Batch generate
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    **gen_kwargs(DO_SAMPLE, TEMPERATURE, TOP_P, GEN_MAX_TOKENS, pad_id),
                )

            new_tokens = out[:, inputs.input_ids.shape[1]:]
            replies = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for (jid, clean_titles, tasks_str, domain, job_ad_title, job_sector_category), reply in zip(meta, replies):
                n_titles = len(clean_titles)

                kept_idx = None
                parse_ok = True
                err = None

                try:
                    drop_idx = parse_drop_indices(reply, n_titles)
                    drop_set = set(drop_idx)
                    kept_idx = [i for i in range(1, n_titles + 1) if i not in drop_set]
                except Exception as e:
                    parse_ok = False
                    err = repr(e)

                # Retry once on parse failure (single-item)
                if not parse_ok:
                    try:
                        rp = build_retry_prompt(tasks_str, clean_titles, domain, job_ad_title, job_sector_category)
                        # Retry: force greedy and small tokens to reduce rambling
                        retry_reply = run_one(rp, RETRY_MAX_TOKENS, do_sample_local=False)
                        drop_idx = parse_drop_indices(retry_reply, n_titles)
                        drop_set = set(drop_idx)
                        kept_idx = [i for i in range(1, n_titles + 1) if i not in drop_set]
                        parse_ok = True
                        err = None
                    except Exception as e2:
                        err = (err or "") + " | retry=" + repr(e2)

                # Final fallback
                if not kept_idx:
                    kept_idx = [1, 2] if n_titles >= 2 else [1]

                # Enforce keep bounds
                kept_idx = kept_idx[:MAX_KEEP]
                if len(kept_idx) < MIN_KEEP and n_titles >= MIN_KEEP:
                    for j in range(1, n_titles + 1):
                        if j not in kept_idx:
                            kept_idx.append(j)
                        if len(kept_idx) >= MIN_KEEP:
                            break
                    kept_idx = kept_idx[:MAX_KEEP]

                kept_set = set(kept_idx)
                drop_idx = [i for i in range(1, n_titles + 1) if i not in kept_set]

                final_titles = [clean_titles[i - 1] for i in kept_idx]
                drop_titles = [clean_titles[i - 1] for i in drop_idx]

                fout.write(
                    json.dumps(
                        {
                            "job_id": jid,
                            "candidates": clean_titles,
                            "drop": drop_titles,
                            "final": final_titles,
                            "parse_ok": parse_ok,
                            "err": err,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"[DONE] {(time.time() - start_wall) / 60:.2f} minutes", flush=True)
    print("[DONE] wrote:", OUT_JSONL, flush=True)

if __name__ == "__main__":
    main()
