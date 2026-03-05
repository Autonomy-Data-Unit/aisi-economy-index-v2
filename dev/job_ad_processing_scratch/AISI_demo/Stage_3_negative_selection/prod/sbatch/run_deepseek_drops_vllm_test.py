# -*- coding: utf-8 -*-
"""
DeepSeek R1 negative-selection drops using vLLM on Isambard (GH200).

Hard requirements this script enforces:
- HF cache points to /projects/public/brics/cache (read-only OK)
- HF dynamic modules cache is writable (HF_MODULES_CACHE)
- XDG cache is writable (XDG_CACHE_HOME)
- Output path is deterministic and stage_3-aligned unless OUT_DIR is provided

Expected NPZ keys:
- job_ids
- titles
- job_ad_title
- job_sector_category
- job_tasks
- domain
- job_description
"""

import os
import json
import time
import datetime as dt
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel


# =============================================================================
# SCHEMA
# =============================================================================
class DropResponse(BaseModel):
    drop: list[int]


# =============================================================================
# UTIL
# =============================================================================
def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def extract_last_json_object(text: str) -> str:
    if not text:
        raise ValueError("Empty reply")

    # Strip <think> if present
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


def normalise_tasks(cur_tasks) -> str:
    if isinstance(cur_tasks, (list, np.ndarray)):
        xs = [str(t).strip() for t in cur_tasks if t is not None and str(t).strip()]
        return ", ".join(xs)[:800]
    return str(cur_tasks).strip()[:800]


# =============================================================================
# PROMPTS (DeepSeek-safe, JSON-only, keep 2–3 default; keep 1 only if very sure)
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

    full_excerpt = (full_ad_text or "").strip()[:700]
    full_block = (
        f"\nFULL AD EXCERPT (tools/duties evidence only):\n{full_excerpt}\n"
        if full_excerpt
        else ""
    )

    # No <think>. Hard JSON-only instruction is repeated.
    return f"""
<|im_start|>system
Return ONLY a valid JSON object with exactly one key "drop". No extra text.
<|im_end|>

<|im_start|>user
Return ONLY a valid JSON object with exactly one key "drop". No prose. No markdown.

TASK
Audit occupation matches. DROP candidates that are NOT functional matches for the job.

KEEP POLICY
There are {n_candidates} candidates (1-based).
- Default: KEEP 2–3 candidates.
- KEEP 1 ONLY if you are certain every other candidate is clearly wrong.
- If more than 3 are valid, drop the most generic ones to fit the 3-candidate cap.
- When in doubt, KEEP rather than DROP if functionally plausible.

RANKING NOTE
Candidates are ranked by cosine similarity (rank 1 = highest cosine). This is a weak prior.
Do NOT assume rank 1 is correct.

JOB CONTEXT (SOURCE OF TRUTH)
Title: {job_ad_title}
Sector: {job_sector_category}
Domain: {domain}
Tasks: {tasks_str}
{full_block}

CANDIDATES (1-based)
{numbered}

ANCHOR (FUNCTION FIRST)
- Identify the functional anchor from title + tasks (function, not seniority).
- You MUST keep the anchor unless core evidence contradicts it.
- Title keyword lock: if the title contains a clear functional keyword and a direct-match candidate exists, you MUST keep it.

GATES
- Manager rule (staff-only): if title does NOT include Manager/Lead/Director/Head/Supervisor, DROP manager/supervisor occupations unless tasks explicitly mention managing STAFF (team, colleagues, direct reports) OR rotas/shifts OR hiring/recruitment OR training/performance reviews OR budgeting/P&L.
  Notes: "manage prisoners/customers/stock/projects" is NOT staff management unless it also mentions managing a team.
  Frontline preference: if a clear frontline occupation matching the title exists (e.g., cleaner, cook, retail salesperson, warehouse worker, builder), prefer it and DROP manager/supervisor roles unless the title itself includes Manager/Lead/Supervisor.

- Inspector rule: DROP inspector/auditor/compliance occupations unless tasks explicitly mention inspection/audit/compliance checks, certification, regulatory reporting, site inspections, or audits.

- IT lock: if title/tasks mention concrete tech (Python/SQL/APIs/systems/cloud), keep relevant IT roles even if domain is non-IT.


FINAL CHECK
- Keep 2–3 by default.
- Keeping only 1 requires clear mismatch for all others.

OUTPUT
Return ONLY JSON: {{"drop":[...]}}
<|im_end|>

<|im_start|>assistant
""".strip()


def build_retry_prompt(
    tasks_str: str,
    clean_titles: list[str],
    domain: str,
    job_ad_title: str,
    job_sector_category: str,
) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(clean_titles))

    return f"""
<|im_start|>system
Return ONLY valid JSON: {{"drop":[...]}}. No extra text.
<|im_end|>

<|im_start|>user
Return ONLY valid JSON with key "drop": {{"drop":[...]}}. No prose. No markdown.

Title: {job_ad_title}
Sector: {job_sector_category}
Domain: {domain}
Tasks: {tasks_str}

Candidates (1-based):
{numbered}

Rules:
- Keep 2–3 by default.
- Keep 1 only if very sure all others are clearly wrong.
- Drop only clearly wrong functions.

Output ONLY JSON: {{"drop":[...]}}
<|im_end|>

<|im_start|>assistant
""".strip()


def parse_drop_indices(reply: str, n_titles: int) -> list[int]:
    jtxt = extract_last_json_object(reply)
    raw = DropResponse.model_validate_json(jtxt).drop

    out: list[int] = []
    for x in raw:
        xi = int(x)
        if 1 <= xi <= n_titles:
            out.append(xi)
    return out


def shard_range(start: int, stop: int, n_rows: int) -> tuple[int, int]:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))

    a0 = max(0, start)
    b0 = min(stop, n_rows)
    n = max(0, b0 - a0)

    s = a0 + (n * rank) // world
    e = a0 + (n * (rank + 1)) // world
    return s, e


def _require_keys(data: np.lib.npyio.NpzFile, keys: Iterable[str]) -> None:
    files = set(data.files)
    missing = [k for k in keys if k not in files]
    if missing:
        raise KeyError(f"NPZ missing keys: {missing}. Found keys: {sorted(files)}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    start_wall = time.time()

    print("[START]", _now(), flush=True)
    print("[ENV] CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

    # --- canonical project paths ---
    project = Path(os.environ.get("PROJECT", "/projects/a5u/adu_dev/aisi-economy-index"))
    stage3_base = project / "aisi_economy_index/store/AISI_demo/stage_3/prod"

    # --- enforce cache discipline (prevents PermissionError on /projects/public/brics/cache/modules) ---
    brics_cache = "/projects/public/brics/cache"
    os.environ.setdefault("HF_HOME", brics_cache)
    os.environ.setdefault("HF_HUB_CACHE", brics_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", brics_cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Writable dynamic modules cache (critical for trust_remote_code models)
    os.environ.setdefault("HF_MODULES_CACHE", str(project / ".hf_modules"))
    _ensure_dir(Path(os.environ["HF_MODULES_CACHE"]))

    # Writable general cache
    os.environ.setdefault("XDG_CACHE_HOME", str(project / ".cache"))
    _ensure_dir(Path(os.environ["XDG_CACHE_HOME"]))

    # --- inputs ---
    if "NPZ_PATH" not in os.environ:
        raise RuntimeError("Missing env var NPZ_PATH")

    npz_path = Path(os.environ["NPZ_PATH"])
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ_PATH does not exist: {npz_path}")

    month_tag = os.environ.get("MONTH_TAG", npz_path.stem)
    embed = os.environ.get("EMBED", "unknown_embed")
    model_id = os.environ.get("MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

    start = _get_env_int("START", 0)
    stop = _get_env_int("STOP", 1000)
    test_only = _get_env_int("TEST_ONLY", 1)
    test_n = _get_env_int("TEST_N", 100)
    chunk_size = _get_env_int("CHUNK_SIZE", 128)

    gen_max_tokens = _get_env_int("GEN_MAX_TOKENS", 512)
    retry_max_tokens = _get_env_int("RETRY_MAX_TOKENS", 160)

    # DeepSeek tends to be chatty at higher temps. Default safer.
    temperature = _get_env_float("TEMPERATURE", 0.3)
    top_p = _get_env_float("TOP_P", 0.95)

    # Policy defaults: keep 2–3, 1 only if very sure (script still enforces MIN/MAX)
    max_keep = _get_env_int("MAX_KEEP", 3)
    min_keep = _get_env_int("MIN_KEEP", 2)

    tp_size = _get_env_int("TP_SIZE", 1)

    # Quantisation: default None (empty -> None). Try fp8 later only if verified.
    quant_raw = os.environ.get("QUANTIZATION", "").strip()
    quantization = quant_raw if quant_raw else None

    # --- output directory (deterministic, stage_3 aligned) ---
    default_out_dir = stage3_base / "llm_negative_selection" / "deepseek" / embed / month_tag
    out_dir = Path(os.environ.get("OUT_DIR", str(default_out_dir)))
    _ensure_dir(out_dir)

    jobid = os.environ.get("SLURM_JOB_ID", "nojid")
    taskid = os.environ.get("SLURM_ARRAY_TASK_ID", "notask")
    rank = os.environ.get("SLURM_PROCID", "0")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_jsonl = out_dir / f"vllm_drop_{month_tag}_{start}_{stop}_job{jobid}_task{taskid}_rank{rank}_{ts}.jsonl"

    print(f"[NPZ] {npz_path}", flush=True)
    print(f"[OUT] {out_jsonl}", flush=True)
    print(f"[MODEL] {model_id}", flush=True)
    print(
        f"[CFG] tp={tp_size} quant={quantization} max_len=8192 gen={gen_max_tokens} retry={retry_max_tokens} chunk={chunk_size}",
        flush=True,
    )
    print(
        f"[CACHE] HF_HOME={os.environ.get('HF_HOME')} HF_MODULES_CACHE={os.environ.get('HF_MODULES_CACHE')}",
        flush=True,
    )
    print(
        f"[POLICY] min_keep={min_keep} max_keep={max_keep} temp={temperature} top_p={top_p}",
        flush=True,
    )

    # --- load data ---
    data = np.load(npz_path, allow_pickle=True)
    _require_keys(
        data,
        [
            "job_ids",
            "titles",
            "job_ad_title",
            "job_sector_category",
            "job_tasks",
            "domain",
            "job_description",
        ],
    )

    job_ids = data["job_ids"]
    candidates = data["titles"]
    job_ad_titles = data["job_ad_title"]
    job_sector_categories = data["job_sector_category"]
    job_tasks = data["job_tasks"]
    domains = data["domain"]
    job_full_ads = data["job_description"]

    n_rows = len(job_ids)
    a, b = shard_range(start, stop, n_rows)

    if test_only:
        b = min(b, a + test_n)

    print(f"[RANGE] {a}:{b} -> {b-a} jobs", flush=True)

    # --- vLLM import and init ---
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        raise RuntimeError(
            "vLLM import failed. You are not running inside the E4S image or vLLM isn't installed. "
            f"Original error: {e}"
        )

    llm_kwargs = dict(
        model=_safe_str(model_id),
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16",
        disable_log_stats=True,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=gen_max_tokens,
        stop=["<|im_end|>"],
    )
    retry_params = SamplingParams(
        temperature=0.0,
        max_tokens=retry_max_tokens,
        stop=["<|im_end|>"],
    )

    n_ok = 0
    n_fail = 0
    n_fallback = 0
    n_empty_titles = 0

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for b0 in range(a, b, chunk_size):
            b1 = min(b0 + chunk_size, b)

            batch_prompts: list[str] = []
            batch_meta: list[tuple[str, list[str], str, str, str, str]] = []

            for i in range(b0, b1):
                jid = str(job_ids[i])

                titles_raw = list(candidates[i]) if candidates[i] is not None else []
                clean_titles = [str(t).strip() for t in titles_raw if t is not None and str(t).strip()]

                if not clean_titles:
                    n_empty_titles += 1
                    fout.write(
                        json.dumps(
                            {
                                "job_id": jid,
                                "candidates": [],
                                "drop": [],
                                "final": [],
                                "parse_ok": True,
                                "err": None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                tasks_str = normalise_tasks(job_tasks[i])

                prompt = build_prompt(
                    tasks_str=tasks_str,
                    clean_titles=clean_titles,
                    domain=str(domains[i]),
                    job_ad_title=str(job_ad_titles[i]),
                    job_sector_category=str(job_sector_categories[i]),
                    full_ad_text=str(job_full_ads[i]) if job_full_ads[i] is not None else "",
                )

                batch_prompts.append(prompt)
                batch_meta.append(
                    (
                        jid,
                        clean_titles,
                        tasks_str,
                        str(domains[i]),
                        str(job_ad_titles[i]),
                        str(job_sector_categories[i]),
                    )
                )

            if not batch_prompts:
                continue

            outputs = llm.generate(batch_prompts, sampling_params)

            for out, meta in zip(outputs, batch_meta):
                jid, clean_titles, tasks_str, dom, title, sector = meta
                n_titles = len(clean_titles)

                text = out.outputs[0].text if out.outputs else ""

                parse_ok = True
                err = None
                kept_idx: list[int] | None = None

                try:
                    drop_idx = parse_drop_indices(text, n_titles)
                    drop_set = set(drop_idx)
                    kept_idx = [x for x in range(1, n_titles + 1) if x not in drop_set]
                except Exception as e:
                    parse_ok = False
                    err = repr(e)

                if not parse_ok:
                    try:
                        rp = build_retry_prompt(tasks_str, clean_titles, dom, title, sector)
                        r = llm.generate([rp], retry_params)
                        r_text = r[0].outputs[0].text if r and r[0].outputs else ""

                        drop_idx = parse_drop_indices(r_text, n_titles)
                        drop_set = set(drop_idx)
                        kept_idx = [x for x in range(1, n_titles + 1) if x not in drop_set]

                        parse_ok = True
                        err = None
                    except Exception as e2:
                        err = f"{err} | retry={repr(e2)}"

                if not kept_idx:
                    n_fallback += 1
                    kept_idx = [1, 2] if n_titles >= 2 else [1]

                kept_idx = kept_idx[:max_keep]

                if len(kept_idx) < min_keep and n_titles >= min_keep:
                    for k in range(1, n_titles + 1):
                        if k not in kept_idx:
                            kept_idx.append(k)
                        if len(kept_idx) >= min_keep:
                            break

                kept_set = set(kept_idx)
                drop_idx = [x for x in range(1, n_titles + 1) if x not in kept_set]

                final_titles = [clean_titles[i - 1] for i in kept_idx]
                drop_titles = [clean_titles[i - 1] for i in drop_idx]

                if parse_ok:
                    n_ok += 1
                else:
                    n_fail += 1

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

    mins = (time.time() - start_wall) / 60.0
    print(
        f"[DONE] minutes={mins:.2f} ok={n_ok} fail={n_fail} fallback={n_fallback} empty_titles={n_empty_titles} out={out_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
