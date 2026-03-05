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
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CONFIG (Isambard)
# =============================================================================
PROJECT = Path("/projects/a5u/adu_dev/aisi-economy-index")
BASE = PROJECT / "aisi_economy_index/store/AISI_demo/stage_3/dev""

NPZ_PATH = Path(os.environ["NPZ_PATH"])
MONTH_TAG = os.environ.get("MONTH_TAG", NPZ_PATH.stem)

HF_MODEL_DIR = PROJECT / "hf_cache" / "models--meta-llama--Meta-Llama-3.1-8B-Instruct"

start_data_key = int(os.environ.get("START", 0))
stop_data_key = int(os.environ.get("STOP", 1_000))

BATCH_SIZE = 128
GEN_MAX_TOKENS = 60
MAX_KEEP = 5

jobid = os.environ.get("SLURM_JOB_ID", "nojid")
taskid = os.environ.get("SLURM_ARRAY_TASK_ID", "notask")

OUT_DIR = BASE / "llm_negative_selection" / MONTH_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_JSONL = OUT_DIR / f"llama_drop_only_{MONTH_TAG}_{start_data_key}_{stop_data_key}_job{jobid}_task{taskid}_{ts}.jsonl"

# =============================================================================
# SCHEMA
# =============================================================================
class DropResponse(BaseModel):
    drop: list[int] | None

# =============================================================================
# JSON EXTRACT
# =============================================================================
_JSON_RE = re.compile(r"\{(?:[^\{}]|(?:\{[^\{}]*\}))*\}", flags=re.DOTALL)

def extract_json_object(text: str) -> str:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON found")
    return m.group(0).strip()

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
    print("[MODEL] snapshot:", snapshot)

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

def s(x) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    return "" if x.lower() == "none" else x

def build_prompt(
    job_desc: str,
    job_tasks: str,
    titles: list[str],
    domain: str,
    job_ad_title: str,
    job_sector_category: str,
    full_ad_text: str,
) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))

    full_excerpt = full_ad_text.strip()[:700] if full_ad_text else ""

    full_block = (
        f"\nFULL AD EXCERPT (use only for concrete duties, tools, licences — ignore fluff):\n{full_excerpt}\n"
        if full_excerpt else ""
    )

    return f"""[INST] You review a short list of candidate occupations.
Your task is to DROP only the candidates that are clearly wrong.
Keep everything else.

Read the job context twice before deciding.

JOB CONTEXT (priority order):
1) Title + Domain (primary anchor)
2) Description + Tasks (evidence)
3) Job Category (weak evidence)
4) Full ad excerpt (weak evidence)

Title: {job_ad_title}
Description: {job_desc}
Tasks/Skills: {job_tasks}
Domain: {domain}
Job Category: {job_sector_category}
{full_block}

RE-READ ANCHOR JOB CONTEXT:
Title: {job_ad_title}
Domain: {domain}

CANDIDATES (1-based)
{numbered}

DECISION RULES

1) Infer the PRIMARY day-to-day function from Title + Domain first.

2) DROP a candidate ONLY if their core work is clearly a different profession.
If there is plausible overlap or uncertainty, KEEP.

3) HARD ANCHOR PROTECTION (never drop if present):
If a candidate matches the obvious occupation implied by Title/Domain/Tasks,
you MUST NOT drop it.

Examples:
- nurse / staff nurse / rgn / rmn -> never drop nurse-family roles
- security officer / guard / patrol / sia -> never drop security guard roles
- developer / software / react / backend / frontend -> never drop software developer roles
- receptionist / front desk / admin assistant -> never drop receptionist/admin family
- plumber/electrician/cable jointing + trainer/assessor -> never drop the trade role itself

4) Licence rule (hard only when explicit):
If a regulated licence clearly defines the role, DROP unrelated licensed families.

5) Do not match on shared nouns alone (engineer, analyst, manager, administrator).
Match on daily outputs and work performed.

6) Seniority is NOT a reason to drop.
Same function at different level is better than different function at same level.

7) IT leakage control:
If Domain is non-IT AND the title lacks explicit IT signals,
DROP IT roles by default UNLESS tasks show concrete technical work
(AWS, SQL, coding, CI/CD, infrastructure, etc).

8) Safety rule:
Most jobs have only a few clearly wrong candidates.
Never drop the role that is the closest match to the title.
Never drop all candidates.

REMEMBER ANCHOR JOB CONTEXT:
Title: {job_ad_title}
Domain: {domain}

OUTPUT
Return ONLY valid JSON:
{{"drop":[2,3,7]}}
[/INST]
"""



def main():
    start_wall = time.time()
    print("[START]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("[ENV] torch:", torch.__version__)
    print("[ENV] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[ENV] gpu:", torch.cuda.get_device_name(0))

    print("[LOAD] npz:", NPZ_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)

    job_ids = data["job_id"]
    short_desc = data["job_desc"]
    tasks_and_skills = data["job_tasks"]
    domains = data["domain"]
    candidates = data["titles"]
    job_ad_titles = data["job_ad_title"]
    job_sector_categories = data["job_sector_category"]
    job_full_ads = data["job_description"]

    n = len(job_ids)
    a = max(0, start_data_key)
    b = min(stop_data_key, n)
    print(f"[RANGE] {a}:{b} (n={n}) -> {b-a} jobs")

    model, tokenizer = load_model()

    with open(OUT_JSONL, "w") as fout:
        for b0 in tqdm(range(a, b, BATCH_SIZE), desc="Batches"):
            b1 = min(b0 + BATCH_SIZE, b)

            prompts, meta = [], []
            for i in range(b0, b1):
                jid = str(job_ids[i])
                titles = candidates[i]
                if not titles:
                    fout.write(json.dumps({"job_id": jid, "candidates": [], "drop": [], "final": []}) + "\n")
                    continue

                p = build_prompt(
                    short_desc[i], tasks_and_skills[i], list(titles), domains[i],
                    job_ad_titles[i], job_sector_categories[i], job_full_ads[i]
                )
                prompts.append(p)
                meta.append((jid, list(titles)))

            if not prompts:
                continue

            texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=GEN_MAX_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)

            replies = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            for (jid, titles), reply in zip(meta, replies):
                drop_idx = []
                try:
                    parsed = DropResponse.model_validate_json(extract_json_object(reply))
                    raw = parsed.drop or []
                    for x in raw:
                        xi = int(x)
                        if 1 <= xi <= len(titles):
                            drop_idx.append(xi)
                    drop_idx = sorted(set(drop_idx))
                except Exception:
                    drop_idx = []

                drop_set = set(drop_idx)
                kept = [t for k, t in enumerate(titles, start=1) if k not in drop_set]
                if not kept:
                    kept = [titles[0]]

                kept = kept[:MAX_KEEP]
                kept_set = set(kept)
                dropped = [t for t in titles if t not in kept_set]

                fout.write(json.dumps({"job_id": jid, "candidates": titles, "drop": dropped, "final": kept}) + "\n")

    print("[DONE] wrote:", OUT_JSONL)
    if torch.cuda.is_available():
        print("Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
        print("Reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")

    end_wall = time.time()
    print("[END]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[RUNTIME] {(end_wall-start_wall)/60:.2f} minutes ({(end_wall-start_wall):.1f} seconds)")

if __name__ == "__main__":
    main()
