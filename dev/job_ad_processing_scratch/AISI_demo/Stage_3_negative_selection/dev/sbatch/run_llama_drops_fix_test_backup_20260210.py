# -*- coding: utf-8 -*-
import os
import re
import json
import datetime
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from pydantic import BaseModel, ValidationError
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
    start, depth, last_obj = None, 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
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
# PROMPT FACTORY
# =============================================================================
def build_prompt(
    job_desc: str,
    tasks_str: str,
    clean_titles: list[str],
    domain: str,
    job_ad_title: str,
    job_sector_category: str,
    full_ad_text: str,
) -> str:
    n_candidates = len(clean_titles)
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(clean_titles))
    
    full_excerpt = full_ad_text.strip()[:700] if full_ad_text else ""
    full_block = f"\nFULL AD EXCERPT (use for concrete tools/duties only):\n{full_excerpt}\n" if full_excerpt else ""

    return f"""
TASK
You are a Senior Labor Market Economist auditing SOC matches. Evaluate candidates and DROP those that are NOT functional matches for the job.

GOAL: THE UP TO 4 TARGET
There are {n_candidates} candidates.
- Your goal is to keep 2 to 4 candidates for every job.
- Assume that at least 2 candidates should normally be KEPT.
- Only keep 1 if you are really sure all others are CLEARLY wrong.
- If more than 4 are valid, drop the most generic ones to fit the 4-candidate cap.

DEFAULT BEHAVIOUR
- When in doubt, KEEP rather than DROP, as long as the candidate is in the correct functional family.
- Actively look for a SECOND and THIRD valid match before dropping.

JOB CONTEXT (SOURCE OF TRUTH)
Title: {job_ad_title}
Sector/Domain: {job_sector_category} | {domain}
Functional Tasks: {tasks_str}
{full_block}

CANDIDATES (1-based)
{numbered}

ANCHOR PROTECTION (FUNCTION FIRST)
1) The ANCHOR is the candidate whose FUNCTION best matches the title’s role type (not seniority, not niche specialism).
2) You MUST KEEP the ANCHOR unless CORE EVIDENCE explicitly contradicts it.
3) TITLE KEYWORD LOCK: If the job title contains a functional keyword (e.g. driver, nurse, electrician, developer, sommelier), and a candidate directly matching that function exists, you MUST keep it.

KEEPING LOGIC (JOB MIX RECALL)
After keeping the ANCHOR, you SHOULD normally keep 2–4 additional candidates.

Keep an additional candidate if ANY apply:
- Same functional family (e.g. sommelier <-> waiter; care worker <-> home health aide).
- Same occupation at a different level (e.g. engineer <-> technologist/technician).
- Tasks partially overlap or describe adjacent duties.
- Do NOT keep roles that are purely generic or cross-sector (e.g. general managers, administrators, laborers) unless the tasks clearly align.
- The role plausibly spans more than one SOC in real labour markets.

HIERARCHY & IT GATES
1) MANAGER RULE
   - If the title does NOT include Manager/Lead/Director, only keep a managerial candidate if tasks explicitly describe staff oversight, rotas, or budgeting.
   - Do NOT keep managerial SOCs based on seniority alone.

2) IT DOMAIN LOCK
   - If the title is explicitly IT, do NOT drop the IT anchor regardless of domain.
   - For non-IT domains, keep tech roles if:
     - tasks mention specific tools (Python, SQL, APIs, etc), OR
     - the title strongly implies a digital/technical function (analyst, systems, platform, data).


RE-ANCHOR (FINAL CHECK)
Title: {job_ad_title}
Tasks: {tasks_str}
Count: {n_candidates} candidates.
You should keep 2–4 candidates.
Keeping only 1 should be rare and requires clear mismatch for all others.

OUTPUT
Return ONLY a valid JSON object with exactly one key: "drop".
Schema:
{{"drop":[...]}}
""".strip()

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
    job_desc = data["job_desc"]
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

            prompts, meta = [], []
            for i in range(b0, b1):
                jid = str(job_ids[i])
                
                # MUST-FIX: Fast Path & Deterministic Indexing
                titles_raw = list(candidates[i]) if candidates[i] is not None else []
                clean_titles = [str(t).strip() for t in titles_raw if t and str(t).strip()]

                if not clean_titles:
                    fout.write(json.dumps({"job_id": jid, "candidates": [], "drop": [], "final": []}) + "\n")
                    continue

                # MUST-FIX: Hardened Task Stringification
                cur_tasks = job_tasks[i]
                if isinstance(cur_tasks, (list, np.ndarray)):
                    tasks_str = ", ".join([str(t).strip() for t in cur_tasks if t and str(t).strip()])
                else:
                    tasks_str = str(cur_tasks).strip()

                p = build_prompt(
                    str(job_desc[i]),
                    tasks_str,
                    clean_titles,
                    str(domains[i]),
                    str(job_ad_titles[i]),
                    str(job_sector_categories[i]),
                    str(job_full_ads[i]) if job_full_ads[i] is not None else "",
                )

                prompts.append(p)
                meta.append((jid, clean_titles))

            if not prompts:
                continue

            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_msg}, {"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True
                ) for p in prompts
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=GEN_MAX_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)

            replies = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            for (jid, clean_titles), reply in zip(meta, replies):
                n_titles = len(clean_titles)
                kept_idx = []

                try:
                    jtxt = extract_last_json_object(reply)
                    raw_drop = DropResponse.model_validate_json(jtxt).drop
                    # Validate indices against clean_titles
                    drop_set = {int(x) for x in raw_drop if 1 <= int(x) <= n_titles}
                    kept_idx = [i for i in range(1, n_titles + 1) if i not in drop_set]
                except:
                    kept_idx = [1]

                # Clamp density & Build partition
                kept_idx = kept_idx[:MAX_KEEP] if kept_idx else [1]
                drop_idx = [i for i in range(1, n_titles + 1) if i not in set(kept_idx)]

                final_titles = [clean_titles[i-1] for i in kept_idx]
                drop_titles = [clean_titles[i-1] for i in drop_idx]

                fout.write(json.dumps({
                    "job_id": jid,
                    "candidates": clean_titles,
                    "drop": drop_titles,
                    "final": final_titles
                }, ensure_ascii=False) + "\n")

    print(f"[DONE] {(time.time()-start_wall)/60:.2f} minutes", flush=True)
    print("[DONE] wrote:", OUT_JSONL, flush=True)

if __name__ == "__main__":
    main()