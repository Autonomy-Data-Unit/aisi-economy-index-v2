# -*- coding: utf-8 -*-
import os
import json
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CONFIG (Isambard)
# =============================================================================
PROJECT = Path("/projects/a5u/adu_dev/aisi-economy-index")
BASE = PROJECT / "aisi_economy_index/store/AISI_demo/stage_3/dev"

NPZ_PATH = Path(os.environ["NPZ_PATH"])
MONTH_TAG = os.environ.get("MONTH_TAG", NPZ_PATH.stem)

HF_MODEL_DIR = PROJECT / "hf_cache" / "models--meta-llama--Meta-Llama-3.1-8B-Instruct"

start_data_key = int(os.environ.get("START", 0))
stop_data_key = int(os.environ.get("STOP", 1_000))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", 80))
MAX_KEEP = int(os.environ.get("MAX_KEEP", 4))

jobid = os.environ.get("SLURM_JOB_ID", "nojid")
taskid = os.environ.get("SLURM_ARRAY_TASK_ID", "notask")

EMBED = os.environ.get("EMBED", "unknown_embed")
OUT_DIR = BASE / "llm_negative_selection" / EMBED / MONTH_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_JSONL = OUT_DIR / (
    f"llama_drop_only_{MONTH_TAG}_{start_data_key}_{stop_data_key}_"
    f"job{jobid}_task{taskid}_{ts}.jsonl"
)

# =============================================================================
# SCHEMA (STRICT)
# =============================================================================
class DropResponse(BaseModel):
    drop: list[int]

# =============================================================================
# JSON EXTRACTION (HARDENED)
# =============================================================================
def extract_last_json_object(text: str) -> str:
    if not text:
        raise ValueError("Empty reply")

    start = None
    depth = 0
    last_obj = None

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

def resolve_snapshot_dir(model_root: Path) -> Path:
    snap_root = model_root / "snapshots"
    if not snap_root.exists():
        raise FileNotFoundError(f"Missing snapshots dir: {snap_root}")
    snaps = [p for p in snap_root.iterdir() if p.is_dir()]
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under: {snap_root}")
    snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0]

def load_model():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    snapshot = resolve_snapshot_dir(HF_MODEL_DIR)
    print("[MODEL] snapshot:", snapshot, flush=True)

    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    mdl.eval()
    return mdl, tok

# =============================================================================
# PROMPT FACTORY (LLAMA CHAT TEMPLATE FRIENDLY)
# Similar structure to your memorised Mistral prompt, but WITHOUT <s>[INST] wrappers.
# The wrappers are added by tokenizer.apply_chat_template.
# =============================================================================
def build_prompt(
    tasks_str: str,
    clean_titles: list[str],
    domain: str,
    job_ad_title: str,
    job_sector_category: str,
    full_ad_text: str,
) -> str:
    n_candidates = len(clean_titles)
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(clean_titles))

    full_excerpt = (full_ad_text or "").strip()[:400]
    full_block = f"\nFULL AD EXCERPT:\n{full_excerpt}\n" if full_excerpt else ""

    body = f"""
Return ONLY a valid JSON object with exactly one key "drop". No extra text.

TASK
Audit occupation matches. DROP candidates that are NOT functional matches for the job.

KEEP POLICY
There are {n_candidates} candidates (1-based).
- Default: KEEP 2 to 3 candidates.
- KEEP 1 ONLY if you are certain every other candidate is clearly wrong.
- If more than 3 are valid, drop the most generic ones to fit the 3-candidate cap.
- When in doubt, KEEP rather than DROP if functionally plausible.

JOB CONTEXT (SOURCE OF TRUTH)
Title: {job_ad_title}
Sector: {job_sector_category}
Domain: {domain}
Tasks: {tasks_str}
{full_block}

CANDIDATES (1-based)
{numbered}

OUTPUT
Return ONLY JSON: {{"drop":[...]}}
""".strip()

    return body

def build_retry_prompt(
    tasks_str: str,
    clean_titles: list[str],
    domain: str,
    job_ad_title: str,
    job_sector_category: str,
) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(clean_titles))

    body = f"""
Return ONLY valid JSON: {{"drop":[...]}}. No extra text.

Title: {job_ad_title}
Sector: {job_sector_category}
Domain: {domain}
Tasks: {tasks_str}

Candidates (1-based):
{numbered}

Rules:
- Keep 2 to 3 by default.
- Keep 1 only if very sure all others are clearly wrong.
- Drop only clearly wrong functions.

Output ONLY JSON: {{"drop":[...]}}
""".strip()

    return body

# =============================================================================
# GENERATION HELPERS
# =============================================================================
def make_chat_text(tokenizer, system_msg: str, user_msg: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )

def generate_batch(model, tokenizer, chat_texts: list[str], max_new_tokens: int) -> list[str]:
    inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    replies = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return replies

def parse_drop_indices(reply: str, n_titles: int) -> tuple[set[int], str]:
    """
    Returns (drop_set, extracted_json_text).
    drop_set is validated to be within 1..n_titles.
    """
    jtxt = extract_last_json_object(reply)
    raw = DropResponse.model_validate_json(jtxt).drop
    drop_set = {int(x) for x in raw if 1 <= int(x) <= n_titles}
    return drop_set, jtxt

# =============================================================================
# MAIN
# =============================================================================
def main():
    start_wall = time.time()
    print("[START]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    print("[LOAD] npz:", NPZ_PATH, flush=True)
    data = np.load(NPZ_PATH, allow_pickle=True)

    job_ids = data["job_ids"]
    candidates = data["titles"]
    job_ad_titles = data["job_ad_title"]
    job_sector_categories = data["job_sector_category"]
    job_tasks = data["job_tasks"]
    domains = data["domain"]
    job_full_ads = data["job_description"]

    n = len(job_ids)
    a = max(0, start_data_key)
    b = min(stop_data_key, n)
    print(f"[RANGE] {a}:{b} -> {b-a} jobs", flush=True)

    model, tokenizer = load_model()

    system_msg = "You are a strict JSON-only classifier. Output ONLY valid JSON matching the schema."

    with open(OUT_JSONL, "w", encoding="utf-8") as fout:
        for b0 in tqdm(range(a, b, BATCH_SIZE), desc="Batches"):
            b1 = min(b0 + BATCH_SIZE, b)

            prompts = []
            meta = []

            for i in range(b0, b1):
                jid = str(job_ids[i])

                titles_raw = list(candidates[i]) if candidates[i] is not None else []
                clean_titles = [str(t).strip() for t in titles_raw if t and str(t).strip()]

                if not clean_titles:
                    fout.write(json.dumps({"job_id": jid, "candidates": [], "drop": [], "final": []}) + "\n")
                    continue

                cur_tasks = job_tasks[i]
                if isinstance(cur_tasks, (list, np.ndarray)):
                    tasks_str = ", ".join([str(t).strip() for t in cur_tasks if t and str(t).strip()])
                else:
                    tasks_str = str(cur_tasks).strip()

                user_prompt = build_prompt(
                    tasks_str=tasks_str,
                    clean_titles=clean_titles,
                    domain=str(domains[i]),
                    job_ad_title=str(job_ad_titles[i]),
                    job_sector_category=str(job_sector_categories[i]),
                    full_ad_text=str(job_full_ads[i]) if job_full_ads[i] is not None else "",
                )

                prompts.append(user_prompt)
                meta.append(
                    {
                        "job_id": jid,
                        "clean_titles": clean_titles,
                        "tasks_str": tasks_str,
                        "domain": str(domains[i]),
                        "job_ad_title": str(job_ad_titles[i]),
                        "job_sector_category": str(job_sector_categories[i]),
                    }
                )

            if not prompts:
                continue

            chat_texts = [make_chat_text(tokenizer, system_msg, p) for p in prompts]
            replies = generate_batch(model, tokenizer, chat_texts, GEN_MAX_TOKENS)

            # First pass parse; collect failures for optional retry
            results = []
            retry_idxs = []

            for k, reply in enumerate(replies):
                m = meta[k]
                clean_titles = m["clean_titles"]
                n_titles = len(clean_titles)

                try:
                    drop_set, jtxt = parse_drop_indices(reply, n_titles)
                    results.append(
                        {
                            "ok": True,
                            "job_id": m["job_id"],
                            "drop_set": drop_set,
                            "reply": reply,
                            "json": jtxt,
                        }
                    )
                except Exception:
                    results.append(
                        {
                            "ok": False,
                            "job_id": m["job_id"],
                            "drop_set": set(),
                            "reply": reply,
                            "json": "",
                        }
                    )
                    retry_idxs.append(k)

            # Optional retry for failures (single extra pass, same batch)
            if retry_idxs:
                retry_prompts = []
                retry_meta = []

                for k in retry_idxs:
                    m = meta[k]
                    retry_user_prompt = build_retry_prompt(
                        tasks_str=m["tasks_str"],
                        clean_titles=m["clean_titles"],
                        domain=m["domain"],
                        job_ad_title=m["job_ad_title"],
                        job_sector_category=m["job_sector_category"],
                    )
                    retry_prompts.append(retry_user_prompt)
                    retry_meta.append((k, m))

                retry_chat_texts = [make_chat_text(tokenizer, system_msg, p) for p in retry_prompts]
                retry_replies = generate_batch(model, tokenizer, retry_chat_texts, GEN_MAX_TOKENS)

                for (orig_k, m), reply in zip(retry_meta, retry_replies):
                    clean_titles = m["clean_titles"]
                    n_titles = len(clean_titles)
                    try:
                        drop_set, jtxt = parse_drop_indices(reply, n_titles)
                        results[orig_k] = {
                            "ok": True,
                            "job_id": m["job_id"],
                            "drop_set": drop_set,
                            "reply": reply,
                            "json": jtxt,
                        }
                    except Exception:
                        # leave as failed, will fall back below
                        results[orig_k]["reply"] = reply

            # Write outputs
            for k, m in enumerate(meta):
                jid = m["job_id"]
                clean_titles = m["clean_titles"]
                n_titles = len(clean_titles)

                drop_set = results[k]["drop_set"] if results[k]["ok"] else set()

                kept_idx = [i for i in range(1, n_titles + 1) if i not in drop_set]
                if not kept_idx:
                    kept_idx = [1]

                kept_idx = kept_idx[:MAX_KEEP]
                drop_idx = [i for i in range(1, n_titles + 1) if i not in set(kept_idx)]

                final_titles = [clean_titles[i - 1] for i in kept_idx]
                drop_titles = [clean_titles[i - 1] for i in drop_idx]

                fout.write(
                    json.dumps(
                        {
                            "job_id": jid,
                            "candidates": clean_titles,
                            "drop": drop_titles,
                            "final": final_titles,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"[DONE] {(time.time() - start_wall) / 60:.2f} minutes", flush=True)
    print("[DONE] wrote:", OUT_JSONL, flush=True)

if __name__ == "__main__":
    main()
