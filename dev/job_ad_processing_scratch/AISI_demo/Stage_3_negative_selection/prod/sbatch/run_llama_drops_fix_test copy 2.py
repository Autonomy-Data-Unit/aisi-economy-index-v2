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
BASE = PROJECT / "aisi_economy_index/store/AISI_demo/stage_3/dev"

NPZ_PATH = Path(os.environ["NPZ_PATH"])
MONTH_TAG = os.environ.get("MONTH_TAG", NPZ_PATH.stem)

HF_MODEL_DIR = PROJECT / "hf_cache" / "models--meta-llama--Meta-Llama-3.1-8B-Instruct"

start_data_key = int(os.environ.get("START", 0))
stop_data_key = int(os.environ.get("STOP", 1_000))

BATCH_SIZE = 128
GEN_MAX_TOKENS = 60
MAX_KEEP = 3

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
         f"\nFULL AD EXCERPT (use only for concrete duties, tools, licences; ignore fluff):\n{full_excerpt}\n"
        if full_excerpt else ""
    )

    return f"""
You are an Occupation Matcher. Your job is to DROP candidates that do NOT fit the job.
Default outcome: keep ONE best match. Keep 2–3 only if tasks clearly show distinct roles (not synonyms).

JOB CONTEXT
Title: {job_ad_title}
Domain: {domain}
Category: {job_sector_category}
Short Desc: {job_desc}
Tasks/Skills: {job_tasks}
{full_block}

CANDIDATES (1-based)
{numbered}

RE-READ ANCHOR JOB CONTEXT:
Title: {job_ad_title}
Domain: {domain}
Tasks/Skills: {job_tasks}

DECISION GUIDELINES
1) Anchor first: prefer the candidate that most directly matches the job title, unless tasks clearly contradict it.
2) Specificity: if both a specialist and a generic parent are present, keep the specialist and drop the generic.
3) Primary role bias: keep one candidate unless tasks prove truly multi-functional work.
4) Non-IT domains: drop Software/Data roles unless there is explicit evidence of building software (coding, pipelines, systems, APIs, models). Using tools (Excel, CRM, SAP) is not enough.
5) Max kept: never keep more than 3.

OUTPUT FORMAT
Return ONLY valid JSON with exactly this key:
{{"drop":[...]}}
""".strip()


def main():
    start_wall = time.time()
    print("[START]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("[CFG] EMBED:", EMBED, flush=True)
    print("[CFG] MONTH_TAG:", MONTH_TAG, flush=True)
    print("[CFG] OUT_DIR:", OUT_DIR, flush=True)
    print("[CFG] OUT_JSONL:", OUT_JSONL, flush=True)

    print("[ENV] torch:", torch.__version__, flush=True)
    print("[ENV] cuda available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("[ENV] gpu:", torch.cuda.get_device_name(0), flush=True)

    print("[LOAD] npz:", NPZ_PATH, flush=True)
    data = np.load(NPZ_PATH, allow_pickle=True)

    # Expect canonical keys from stage3-prep
    job_ids = data["job_ids"]
    job_desc = data["job_desc"]
    job_tasks = data["job_tasks"]
    domains = data["domain"]
    candidates = data["titles"]
    job_ad_titles = data["job_ad_title"]
    job_sector_categories = data["job_sector_category"]
    job_full_ads = data["job_description"]

    n = len(job_ids)
    a = max(0, start_data_key)
    b = min(stop_data_key, n)
    print(f"[RANGE] {a}:{b} (n={n}) -> {b-a} jobs", flush=True)

    model, tokenizer = load_model()

    with open(OUT_JSONL, "w", encoding="utf-8") as fout:
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
                    job_desc[i],
                    job_tasks[i],
                    list(titles),
                    str(domains[i]),
                    str(job_ad_titles[i]),
                    str(job_sector_categories[i]),
                    str(job_full_ads[i]) if job_full_ads[i] is not None else "",
                )

                if i == b0:  # first item in each batch
                    print("[DBG] job_id:", jid, flush=True)
                    print("[DBG] title:", repr(job_ad_titles[i]), flush=True)
                    print("[DBG] domain:", repr(domains[i]), flush=True)
                    print("[DBG] category:", repr(job_sector_categories[i]), flush=True)
                    print("[DBG] desc_head:", repr(str(job_desc[i])[:80]), flush=True)
                    print("[DBG] tasks_head:", repr(str(job_tasks[i])[:80]), flush=True)
                    
                prompts.append(p)
                meta.append((jid, list(titles)))

            if not prompts:
                continue

            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
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
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=GEN_MAX_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            replies = tokenizer.batch_decode(
                out[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

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

                fout.write(
                    json.dumps(
                        {"job_id": jid, "candidates": titles, "drop": dropped, "final": kept},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print("[DONE] wrote:", OUT_JSONL, flush=True)
    if torch.cuda.is_available():
        print("Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB", flush=True)
        print("Reserved:", torch.cuda.memory_reserved() / 1024**3, "GB", flush=True)

    end_wall = time.time()
    print("[END]", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(f"[RUNTIME] {(end_wall-start_wall)/60:.2f} minutes ({(end_wall-start_wall):.1f} seconds)", flush=True)


if __name__ == "__main__":
    main()
