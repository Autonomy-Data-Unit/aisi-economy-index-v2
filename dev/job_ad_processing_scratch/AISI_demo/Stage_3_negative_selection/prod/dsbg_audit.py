# -*- coding: utf-8 -*-
"""
DeepSeek + BGE only
Stage-3 QA: drift evaluator + comprehensive audit + second-pass rerun selector.

Writes under ./reports/deepseek_bge_audit/<timestamp>/:
- <ts>_DRIFT_REPORT.txt
- <ts>_AUDIT_SUMMARY.json
- <ts>_ANOMALIES.jsonl
- <ts>_RANDOM_SAMPLES.jsonl
- <ts>_SECOND_PASS_RERUN.jsonl
- <ts>_SECOND_PASS_JOB_IDS.json

No plots. Pure QA + rerun list.
"""

from __future__ import annotations

import json
import re
import random
import hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------
# Regex / heuristics
# ---------------------------------------------------------------------

_IT_OCC_LABEL_PATTERNS = [
    r"\bsoftware\b", r"\bdeveloper\b", r"\bprogrammer\b",
    r"\bdata\b", r"\bmachine\s*learning\b", r"\bml\b", r"\bai\b",
    r"\bcloud\b", r"\bdevops\b", r"\bsite\s*reliability\b", r"\bsre\b",
    r"\bcyber\b", r"\binformation\s*security\b", r"\bsecurity\b",
    r"\bnetwork\b", r"\bsystems?\b", r"\bdatabase\b",
    r"\bweb\b", r"\bfull\s*stack\b", r"\bbackend\b", r"\bfrontend\b",
    r"\bit\b",
]
_IT_OCC_RE = re.compile("|".join(_IT_OCC_LABEL_PATTERNS), flags=re.I)

_IT_CTX_DOMAIN_PATTERNS = [
    r"\bit\b", r"\btechnology\b", r"\bsoftware\b", r"\bdata\b", r"\bcyber\b", r"\bcloud\b",
    r"\bengineering\b", r"\bdeveloper\b", r"\bdevops\b",
]
_IT_CTX_DOMAIN_RE = re.compile("|".join(_IT_CTX_DOMAIN_PATTERNS), flags=re.I)

_IT_CTX_TASK_PATTERNS = [
    r"\bpython\b", r"\bsql\b", r"\bscala\b", r"\bjava\b", r"\bc\+\+\b",
    r"\bjavascript\b", r"\btypescript\b", r"\bnode\b", r"\breact\b", r"\bangular\b",
    r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bkubernetes\b", r"\bdocker\b",
    r"\bterraform\b", r"\bansible\b", r"\bci\/cd\b", r"\bpipeline\b",
    r"\bapi\b", r"\bmicroservices?\b", r"\bbackend\b", r"\bfrontend\b",
    r"\bnetwork\b", r"\bserver\b", r"\blinux\b", r"\bwindows\s*server\b",
    r"\bsiem\b", r"\bpenetration\b", r"\bvulnerability\b",
]
_IT_CTX_TASK_RE = re.compile("|".join(_IT_CTX_TASK_PATTERNS), flags=re.I)

_MANAGER_OCC_LABEL_RE = re.compile(
    r"\b(manager|managers|director|management|supervisor|supervisors|head|chief)\b",
    flags=re.I,
)
_MANAGER_TITLE_HINTS_RE = re.compile(r"\b(manager|director|head|lead|principal|chief)\b", flags=re.I)

_MANAGER_TASK_PATTERNS_STRONG = [
    r"\bline\s*manage\b", r"\bmanage\b.*\bteam\b", r"\bpeople\s*management\b",
    r"\bhiring\b", r"\brecruit(ing|ment)\b", r"\bperformance\b.*\breview\b",
    r"\bbudget\b", r"\bp&l\b", r"\bfinancial\s*responsibilit(y|ies)\b",
    r"\bstrategy\b", r"\broadmap\b", r"\bgovernance\b",
    r"\bokrs?\b", r"\bkpi(s)?\b", r"\bstakeholders?\b",
    r"\bresource\b.*\bplan\b", r"\bcapacity\b.*\bplan\b",
]
_MANAGER_TASK_RE_STRONG = re.compile("|".join(_MANAGER_TASK_PATTERNS_STRONG), flags=re.I)

_NON_MANAGER_TITLE_RE = re.compile(
    r"\b(assistant|technician|officer|coordinator|associate|executive)\b",
    flags=re.I,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def dsbg_clean_str(x, fallback: str) -> str:
    if x is None:
        return fallback

    if isinstance(x, (bytes, bytearray, np.bytes_)):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            x = str(x)

    if isinstance(x, np.generic):
        try:
            x = x.item()
        except Exception:
            x = str(x)

    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return fallback
    return s


def dsbg_tasks_to_str(t) -> str:
    if t is None:
        return ""
    if isinstance(t, (list, tuple, np.ndarray)):
        parts = []
        for x in t:
            sx = dsbg_clean_str(x, "")
            if sx:
                parts.append(sx)
        return ", ".join(parts)
    return dsbg_clean_str(t, "")


def dsbg_truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n].rstrip() + " ...[truncated]"


def dsbg_occ_is_it(label: str) -> bool:
    return bool(_IT_OCC_RE.search(label or ""))


def dsbg_occ_is_managerial(label: str) -> bool:
    return bool(_MANAGER_OCC_LABEL_RE.search(label or ""))


def dsbg_ctx_is_it(domain: str, title: str, tasks: str, desc: str) -> bool:
    blob = " ".join([domain or "", title or "", tasks or "", desc or ""])
    return bool(_IT_CTX_DOMAIN_RE.search(blob) or _IT_CTX_TASK_RE.search(blob))


def dsbg_ctx_is_managerial_strong(title: str, tasks: str, desc: str) -> bool:
    t = title or ""
    blob = " ".join([tasks or "", desc or ""])

    if _MANAGER_TITLE_HINTS_RE.search(t):
        return True

    if _NON_MANAGER_TITLE_RE.search(t):
        return bool(_MANAGER_TASK_RE_STRONG.search(blob))

    return bool(_MANAGER_TASK_RE_STRONG.search(blob))


def dsbg_list_probable_full_ad_keys(files: set[str]) -> list[str]:
    out = []
    for k in ["job_description", "full_ad", "description", "ad_text", "job_text", "text"]:
        if k in files:
            out.append(k)
    return out


def dsbg_sha256_text(s: str) -> str:
    s = (s or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(s).hexdigest()


def dsbg_hist_percentiles(hist: Counter, percentiles: list[float]) -> dict[str, int]:
    """
    Percentiles from a discrete histogram.
    percentiles are 0-100.
    """
    total = sum(hist.values())
    if total <= 0:
        return {f"p{int(p):02d}": 0 for p in percentiles}

    keys = sorted(hist.keys())
    cum = 0
    out = {}
    targets = [(p, (p / 100.0) * total) for p in percentiles]
    ti = 0
    for k in keys:
        cum += hist[k]
        while ti < len(targets) and cum >= targets[ti][1]:
            p = targets[ti][0]
            out[f"p{int(p):02d}"] = int(k)
            ti += 1
    while ti < len(targets):
        p = targets[ti][0]
        out[f"p{int(p):02d}"] = int(keys[-1])
        ti += 1
    return out


def dsbg_normalise_list_of_str(x) -> tuple[list[str], list[str]]:
    """
    Returns (cleaned_list, schema_errors)
    """
    errs = []
    if x is None:
        return [], errs

    if isinstance(x, str):
        # upstream sometimes writes a single string
        return [dsbg_clean_str(x, "").strip()], ["field_was_str_not_list"]

    if not isinstance(x, list):
        return [], [f"field_not_list:{type(x).__name__}"]

    out = []
    for v in x:
        sv = dsbg_clean_str(v, "").strip()
        if sv:
            out.append(sv)
    return out, errs


def dsbg_iter_jsonl_rows(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# ---------------------------------------------------------------------
# NPZ loader
# ---------------------------------------------------------------------

def dsbg_load_npz_context(npz_path: Path) -> dict[str, dict]:
    if not npz_path.exists():
        return {}

    with np.load(npz_path, allow_pickle=True) as z:
        files = set(z.files)
        job_id_key = next((k for k in ("job_ids", "job_id") if k in files), None)
        if job_id_key is None:
            return {}

        jids = [dsbg_clean_str(x, "").strip() for x in z[job_id_key].tolist()]

        def get_col(key, default=None):
            if key not in files:
                return [default] * len(jids)
            arr = z[key]
            return arr.tolist() if hasattr(arr, "tolist") else list(arr)

        titles = get_col("job_ad_title", default=None)
        domains = get_col("domain", default=None)
        sectors = get_col("job_sector_category", default=None)
        job_descs = get_col("job_desc", default=None)
        job_tasks = get_col("job_tasks", default=None)

        full_ad_keys = dsbg_list_probable_full_ad_keys(files)
        full_ads = [None] * len(jids)
        used_full_key = None
        if full_ad_keys:
            used_full_key = full_ad_keys[0]
            full_ads = get_col(used_full_key, default=None)

    out: dict[str, dict] = {}
    for i, jid in enumerate(jids):
        if not jid or jid.lower() in ("nan", "none"):
            continue

        title = dsbg_clean_str(titles[i], "N/A")
        domain = dsbg_clean_str(domains[i], "UNKNOWN")
        sector = dsbg_clean_str(sectors[i], "UNKNOWN")
        desc = dsbg_clean_str(job_descs[i], "")
        tasks = dsbg_tasks_to_str(job_tasks[i])

        full_ad_raw = ""
        if i < len(full_ads):
            full_ad_raw = dsbg_clean_str(full_ads[i], "")

        full_text = full_ad_raw or desc or tasks or title

        out[jid] = {
            "job_id": jid,
            "title": title,
            "domain": domain,
            "sector": sector,
            "desc": desc,
            "tasks": tasks,
            "full_ad_raw": full_ad_raw,
            "full_text": full_text,
            "npz_full_ad_key": used_full_key or "",
        }

    return out


def dsbg_npz_for_month(npz_base_dir: Path, month: str) -> Path:
    return npz_base_dir / f"{month}.npz"


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------

def dsbg_reservoir_add(reservoir: list[dict], item: dict, k: int, seen: int, rng: random.Random) -> None:
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.randrange(seen)
    if j < k:
        reservoir[j] = item


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------

@dataclass
class MonthStats:
    n_jobs: int = 0
    sum_before: int = 0
    sum_after: int = 0
    before_hist: Counter = None
    after_hist: Counter = None

    empty_final: int = 0
    final_gt_max: int = 0
    final_has_dupes: int = 0
    final_not_subset: int = 0

    missing_context: int = 0
    ctx_full_text_is_title_only: int = 0

    it_drift_jobs: int = 0
    mgr_drift_jobs: int = 0

    selected_for_rerun: int = 0

    def __post_init__(self):
        if self.before_hist is None:
            self.before_hist = Counter()
        if self.after_hist is None:
            self.after_hist = Counter()


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def dsbg_build_audit_report_and_pruning_lists(
    jsonl_base_dir: Path,
    npz_base_dir: Path,
    *,
    llm_model: str,
    prompt_text: str,
    jsonl_glob: str = "*.jsonl",
    reports_subdir: str = "reports/deepseek_bge_audit",
    trunc_ctx_fields: int = 1600,
    sample_trunc: int = 650,
    max_final_expected: int = 3,
    random_seed: int = 7,
    random_n_any: int = 20,
    random_n_stratum: int = 12,
    select_for_rerun_flags: tuple[str, ...] = ("it_drift", "managerial_drift", "empty_final", "final_not_subset", "missing_context"),
) -> dict:
    """
    Comprehensive QA + drift + rerun list builder.

    select_for_rerun_flags controls what goes to SECOND_PASS outputs.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.cwd() / reports_subdir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / f"{ts}_DRIFT_REPORT.txt"
    summary_path = out_dir / f"{ts}_AUDIT_SUMMARY.json"
    anomalies_path = out_dir / f"{ts}_ANOMALIES.jsonl"
    random_path = out_dir / f"{ts}_RANDOM_SAMPLES.jsonl"
    rerun_jsonl_path = out_dir / f"{ts}_SECOND_PASS_RERUN.jsonl"
    rerun_jobids_path = out_dir / f"{ts}_SECOND_PASS_JOB_IDS.json"

    if not jsonl_base_dir.exists():
        raise FileNotFoundError(f"jsonl_base_dir does not exist: {jsonl_base_dir}")
    if not npz_base_dir.exists():
        raise FileNotFoundError(f"npz_base_dir does not exist: {npz_base_dir}")

    month_dirs = sorted([p for p in jsonl_base_dir.iterdir() if p.is_dir()])
    if not month_dirs:
        raise RuntimeError(f"No month folders under: {jsonl_base_dir}")

    rng = random.Random(random_seed)

    prompt_hash = dsbg_sha256_text(prompt_text.strip())

    # Global metrics
    global_stats = MonthStats()
    per_month: dict[str, MonthStats] = defaultdict(MonthStats)

    kept_titles = Counter()
    it_kept_labels = 0
    total_kept_labels = 0

    anomaly_counts = Counter()
    schema_counts = Counter()

    # Sampling reservoirs
    samples_any: list[dict] = []
    samples_by_stratum: dict[str, list[dict]] = {
        "drift": [],
        "empty_final": [],
        "final_not_subset": [],
        "missing_context": [],
    }
    seen_any = 0
    seen_stratum = Counter()

    # Cache NPZ context per month
    ctx_cache: dict[str, dict[str, dict]] = {}

    # Second pass outputs
    compact: dict[str, dict] = {}
    rerun_rows_written = 0

    # Stream writers
    anomalies_f = anomalies_path.open("w", encoding="utf-8")
    rerun_f = rerun_jsonl_path.open("w", encoding="utf-8")

    def write_anomaly(row: dict) -> None:
        anomalies_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_rerun(row: dict) -> None:
        nonlocal rerun_rows_written
        rerun_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rerun_rows_written += 1

    try:
        for month_dir in month_dirs:
            month = month_dir.name
            jsonl_files = sorted(month_dir.glob(jsonl_glob))
            if not jsonl_files:
                continue

            if month not in ctx_cache:
                npz_path = dsbg_npz_for_month(npz_base_dir, month)
                ctx_cache[month] = dsbg_load_npz_context(npz_path) if npz_path.exists() else {}

            ctx_lookup = ctx_cache[month]
            mstats = per_month[month]

            for jp in jsonl_files:
                for obj in dsbg_iter_jsonl_rows(jp):
                    jid = dsbg_clean_str(obj.get("job_id", ""), "").strip()
                    if not jid:
                        schema_counts["missing_job_id"] += 1
                        continue

                    cands_raw = obj.get("candidates", [])
                    final_raw = obj.get("final", [])
                    drop_raw = obj.get("drop", [])

                    candidates, c_errs = dsbg_normalise_list_of_str(cands_raw)
                    final, f_errs = dsbg_normalise_list_of_str(final_raw)
                    if c_errs:
                        schema_counts["candidates_type"] += 1
                    if f_errs:
                        schema_counts["final_type"] += 1

                    # Drop can be anything; keep raw but record schema weirdness
                    if drop_raw is None:
                        drop = []
                    elif isinstance(drop_raw, list):
                        drop = drop_raw
                    else:
                        drop = [drop_raw]
                        schema_counts["drop_type"] += 1

                    b_len = len(candidates) if candidates else len(final)
                    a_len = len(final)

                    # Update stats
                    for st in (global_stats, mstats):
                        st.n_jobs += 1
                        st.sum_before += b_len
                        st.sum_after += a_len
                        st.before_hist[b_len] += 1
                        st.after_hist[a_len] += 1

                    if a_len == 0:
                        global_stats.empty_final += 1
                        mstats.empty_final += 1

                    if a_len > max_final_expected:
                        global_stats.final_gt_max += 1
                        mstats.final_gt_max += 1

                    if len(set(final)) != len(final) and a_len > 1:
                        global_stats.final_has_dupes += 1
                        mstats.final_has_dupes += 1

                    final_not_subset = False
                    if candidates and final:
                        cand_set = set(candidates)
                        final_not_subset = any(x not in cand_set for x in final)
                        if final_not_subset:
                            global_stats.final_not_subset += 1
                            mstats.final_not_subset += 1

                    # Kept titles share
                    for t in final:
                        kept_titles[t] += 1
                        total_kept_labels += 1
                        if dsbg_occ_is_it(t):
                            it_kept_labels += 1

                    # Context lookup
                    ctx = ctx_lookup.get(jid)
                    missing_ctx = ctx is None
                    if missing_ctx:
                        global_stats.missing_context += 1
                        mstats.missing_context += 1
                    else:
                        # full_text is title only means all upstream text fields were empty
                        if (ctx.get("full_text", "") or "").strip() == (ctx.get("title", "") or "").strip():
                            global_stats.ctx_full_text_is_title_only += 1
                            mstats.ctx_full_text_is_title_only += 1

                    # Drift detection (requires context)
                    it_drift = False
                    mgr_drift = False
                    if ctx:
                        ctx_it = dsbg_ctx_is_it(ctx["domain"], ctx["title"], ctx["tasks"], ctx["desc"])
                        ctx_mgr = dsbg_ctx_is_managerial_strong(ctx["title"], ctx["tasks"], ctx["desc"])
                        kept_has_it = any(dsbg_occ_is_it(x) for x in final)
                        kept_has_mgr = any(dsbg_occ_is_managerial(x) for x in final)
                        it_drift = (not ctx_it) and kept_has_it
                        mgr_drift = (not ctx_mgr) and kept_has_mgr
                        if it_drift:
                            global_stats.it_drift_jobs += 1
                            mstats.it_drift_jobs += 1
                        if mgr_drift:
                            global_stats.mgr_drift_jobs += 1
                            mstats.mgr_drift_jobs += 1

                    # Build a compact row for samples/anomalies/rerun
                    ctx_title = ctx["title"] if ctx else "N/A"
                    ctx_domain = ctx["domain"] if ctx else "UNKNOWN"
                    ctx_sector = ctx["sector"] if ctx else "UNKNOWN"
                    ctx_desc = dsbg_truncate(ctx["desc"], trunc_ctx_fields) if ctx else ""
                    ctx_tasks = dsbg_truncate(ctx["tasks"], trunc_ctx_fields) if ctx else ""
                    ctx_full = dsbg_truncate(ctx["full_text"], trunc_ctx_fields) if ctx else ""

                    flags = {
                        "it_drift": bool(it_drift),
                        "managerial_drift": bool(mgr_drift),
                        "empty_final": bool(a_len == 0),
                        "final_gt_max": bool(a_len > max_final_expected),
                        "final_has_dupes": bool(len(set(final)) != len(final) and a_len > 1),
                        "final_not_subset": bool(final_not_subset),
                        "missing_context": bool(missing_ctx),
                        "ctx_full_text_is_title_only": bool((ctx_full.strip() == (ctx_title or "").strip()) if ctx else False),
                    }

                    # Random sampling (reservoir) across all jobs
                    seen_any += 1
                    sample_row = {
                        "job_id": jid,
                        "month": month,
                        "flags": flags,
                        "title": ctx_title,
                        "domain": ctx_domain,
                        "sector": ctx_sector,
                        "desc": dsbg_truncate(ctx_desc, sample_trunc),
                        "tasks": dsbg_truncate(ctx_tasks, sample_trunc),
                        "full_ad": dsbg_truncate(ctx_full, sample_trunc),
                        "kept": final[:],
                        "candidates": candidates[:12],
                    }
                    dsbg_reservoir_add(samples_any, sample_row, random_n_any, seen_any, rng)

                    # Stratified sampling
                    if flags["it_drift"] or flags["managerial_drift"]:
                        seen_stratum["drift"] += 1
                        dsbg_reservoir_add(samples_by_stratum["drift"], sample_row, random_n_stratum, seen_stratum["drift"], rng)
                    if flags["empty_final"]:
                        seen_stratum["empty_final"] += 1
                        dsbg_reservoir_add(samples_by_stratum["empty_final"], sample_row, random_n_stratum, seen_stratum["empty_final"], rng)
                    if flags["final_not_subset"]:
                        seen_stratum["final_not_subset"] += 1
                        dsbg_reservoir_add(samples_by_stratum["final_not_subset"], sample_row, random_n_stratum, seen_stratum["final_not_subset"], rng)
                    if flags["missing_context"]:
                        seen_stratum["missing_context"] += 1
                        dsbg_reservoir_add(samples_by_stratum["missing_context"], sample_row, random_n_stratum, seen_stratum["missing_context"], rng)

                    # Anomalies stream
                    is_anomaly = (
                        flags["empty_final"]
                        or flags["final_gt_max"]
                        or flags["final_has_dupes"]
                        or flags["final_not_subset"]
                        or flags["missing_context"]
                        or flags["ctx_full_text_is_title_only"]
                    )
                    if is_anomaly:
                        for k, v in flags.items():
                            if v and k not in ("it_drift", "managerial_drift"):
                                anomaly_counts[k] += 1

                        write_anomaly({
                            "job_id": jid,
                            "month": month,
                            "flags": flags,
                            "title": ctx_title,
                            "domain": ctx_domain,
                            "sector": ctx_sector,
                            "kept": final,
                            "candidates": candidates,
                            "drop": drop,
                            "desc": dsbg_truncate(ctx_desc, sample_trunc),
                            "tasks": dsbg_truncate(ctx_tasks, sample_trunc),
                            "full_ad": dsbg_truncate(ctx_full, sample_trunc),
                            "source_file": str(jp),
                        })

                    # Second pass rerun selection
                    selected = any(flags.get(k, False) for k in select_for_rerun_flags)
                    if selected:
                        global_stats.selected_for_rerun += 1
                        mstats.selected_for_rerun += 1

                        rerun_row = {
                            "job_id": jid,
                            "month": month,
                            "flags": {k: bool(v) for k, v in flags.items() if v},
                            "title": ctx_title,
                            "domain": ctx_domain,
                            "sector": ctx_sector,
                            "desc": ctx_desc,
                            "tasks": ctx_tasks,
                            "full_ad": ctx_full,
                            "kept": final,
                            "candidates": candidates,
                            "drop": drop,
                        }
                        write_rerun(rerun_row)

                        # Dedupe job_ids + merge candidates + flags
                        if jid not in compact:
                            compact[jid] = {
                                "job_id": jid,
                                "month": month,
                                "flags": {k: bool(v) for k, v in flags.items() if v},
                                "candidates": sorted(set(candidates)),
                                "kept": final,
                                "title": ctx_title,
                                "domain": ctx_domain,
                                "sector": ctx_sector,
                            }
                        else:
                            for fk, fv in flags.items():
                                if fv:
                                    compact[jid]["flags"][fk] = True
                            compact[jid]["candidates"] = sorted(set(compact[jid]["candidates"]) | set(candidates))

    finally:
        anomalies_f.close()
        rerun_f.close()

    # Write random samples
    with random_path.open("w", encoding="utf-8") as f:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "random_seed": random_seed,
            "samples_any": samples_any,
            "samples_by_stratum": samples_by_stratum,
        }
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))

    # Summary stats
    def summarise_stats(st: MonthStats) -> dict:
        avg_before = (st.sum_before / st.n_jobs) if st.n_jobs else 0.0
        avg_after = (st.sum_after / st.n_jobs) if st.n_jobs else 0.0
        drop_rate = 1.0 - (avg_after / avg_before) if avg_before > 0 else 0.0
        return {
            "jobs": st.n_jobs,
            "avg_candidates_before": avg_before,
            "avg_candidates_after": avg_after,
            "drop_rate": drop_rate,
            "empty_final_percent": 100.0 * st.empty_final / max(st.n_jobs, 1),
            "final_gt_max_percent": 100.0 * st.final_gt_max / max(st.n_jobs, 1),
            "final_has_dupes_percent": 100.0 * st.final_has_dupes / max(st.n_jobs, 1),
            "final_not_subset_percent": 100.0 * st.final_not_subset / max(st.n_jobs, 1),
            "missing_context_percent": 100.0 * st.missing_context / max(st.n_jobs, 1),
            "ctx_full_text_is_title_only_percent": 100.0 * st.ctx_full_text_is_title_only / max(st.n_jobs, 1),
            "it_drift_jobs": st.it_drift_jobs,
            "it_drift_percent": 100.0 * st.it_drift_jobs / max(st.n_jobs, 1),
            "managerial_drift_jobs": st.mgr_drift_jobs,
            "managerial_drift_percent": 100.0 * st.mgr_drift_jobs / max(st.n_jobs, 1),
            "selected_for_rerun_jobs": st.selected_for_rerun,
            "selected_for_rerun_percent": 100.0 * st.selected_for_rerun / max(st.n_jobs, 1),
            "before_count_percentiles": dsbg_hist_percentiles(st.before_hist, [5, 25, 50, 75, 95]),
            "after_count_percentiles": dsbg_hist_percentiles(st.after_hist, [5, 25, 50, 75, 95]),
        }

    global_summary = summarise_stats(global_stats)
    per_month_summary = {m: summarise_stats(s) for m, s in sorted(per_month.items(), key=lambda x: x[0])}

    it_label_share = it_kept_labels / max(total_kept_labels, 1)

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "llm_model": llm_model,
        "prompt_hash_sha256": prompt_hash,
        "jsonl_base_dir": str(jsonl_base_dir),
        "npz_base_dir": str(npz_base_dir),
        "select_for_rerun_flags": list(select_for_rerun_flags),
        "max_final_expected": max_final_expected,
        "schema_counts": dict(schema_counts),
        "anomaly_counts": dict(anomaly_counts),
        "it_kept_label_share": it_label_share,
        "global": global_summary,
        "per_month": per_month_summary,
        "top_kept_occupations_top25": kept_titles.most_common(25),
        "outputs": {
            "report_txt": str(report_path),
            "audit_summary_json": str(summary_path),
            "anomalies_jsonl": str(anomalies_path),
            "random_samples_json": str(random_path),
            "second_pass_rerun_jsonl": str(rerun_jsonl_path),
            "second_pass_job_ids_json": str(rerun_jobids_path),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Second pass job ids json
    rerun_jobids_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "llm_model": llm_model,
                "prompt_hash_sha256": prompt_hash,
                "jsonl_base_dir": str(jsonl_base_dir),
                "npz_base_dir": str(npz_base_dir),
                "total_jobs_parsed": global_stats.n_jobs,
                "selected_unique_job_ids": len(compact),
                "selected_rows": rerun_rows_written,
                "job_ids": list(compact.values()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Text report (human readable)
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("DEEPSEEK + BGE AUDIT REPORT (DRIFT + INTEGRITY + RANDOM QA)")
    lines.append("=" * 90)
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"llm_model: {llm_model}")
    lines.append(f"prompt_hash_sha256: {prompt_hash}")
    lines.append(f"cwd: {Path.cwd()}")
    lines.append(f"jsonl_base_dir: {jsonl_base_dir}")
    lines.append(f"npz_base_dir: {npz_base_dir}")
    lines.append(f"out_dir: {out_dir}")
    lines.append(f"jsonl_glob: {jsonl_glob}")
    lines.append("")

    lines.append("GLOBAL METRICS")
    for k, v in global_summary.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append(f"it_kept_label_share: {it_label_share:.4f}")
    lines.append("")

    lines.append("SCHEMA COUNTS")
    for k, v in sorted(schema_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("ANOMALY COUNTS (non drift)")
    for k, v in sorted(anomaly_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("PER MONTH SUMMARY (jobs, empty_final%, missing_ctx%, subset_violation%)")
    for m, s in per_month_summary.items():
        lines.append(
            f"{m}: jobs={s['jobs']}, empty_final%={s['empty_final_percent']:.2f}, "
            f"missing_ctx%={s['missing_context_percent']:.2f}, final_not_subset%={s['final_not_subset_percent']:.2f}, "
            f"it_drift%={s['it_drift_percent']:.2f}, mgr_drift%={s['managerial_drift_percent']:.2f}"
        )
    lines.append("")

    lines.append("TOP KEPT OCCUPATIONS (top 25)")
    for title, count in kept_titles.most_common(25):
        lines.append(f"{count:>6}  {title}")
    lines.append("")

    lines.append("PROMPT USED (truncated to 2000 chars)")
    lines.append("-" * 90)
    lines.append(dsbg_truncate(prompt_text.strip(), 2000))
    lines.append("-" * 90)
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append(f"audit_summary_json: {summary_path}")
    lines.append(f"anomalies_jsonl: {anomalies_path}")
    lines.append(f"random_samples_json: {random_path}")
    lines.append(f"second_pass_rerun_jsonl: {rerun_jsonl_path}")
    lines.append(f"second_pass_job_ids_json: {rerun_jobids_path}")
    lines.append("")

    lines.append("RANDOM QA SAMPLES (any)")
    for s in samples_any[: min(len(samples_any), 10)]:
        lines.append("")
        lines.append(f"job_id: {s['job_id']} | month={s['month']}")
        lines.append(f"flags: {s['flags']}")
        lines.append(f"title: {s['title']}")
        lines.append(f"domain: {s['domain']} | sector: {s['sector']}")
        if s.get("desc"):
            lines.append(f"desc: {s['desc']}")
        if s.get("tasks"):
            lines.append(f"tasks: {s['tasks']}")
        if s.get("full_ad"):
            lines.append(f"full_ad: {s['full_ad']}")
        lines.append(f"kept: {s['kept']}")
        lines.append(f"candidates: {s['candidates']}")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Wrote:")
    print(f"  report:   {report_path}")
    print(f"  summary:  {summary_path}")
    print(f"  anomalies:{anomalies_path}")
    print(f"  random:   {random_path}")
    print(f"  rerun:    {rerun_jsonl_path}")
    print(f"  job_ids:  {rerun_jobids_path}")

    return {
        "out_dir": str(out_dir),
        "report_txt": str(report_path),
        "audit_summary_json": str(summary_path),
        "anomalies_jsonl": str(anomalies_path),
        "random_samples_json": str(random_path),
        "second_pass_rerun_jsonl": str(rerun_jsonl_path),
        "second_pass_job_ids_json": str(rerun_jobids_path),
        "total_jobs_parsed": global_stats.n_jobs,
        "selected_unique_job_ids": len(compact),
        "selected_rows": rerun_rows_written,
        "it_kept_label_share": it_label_share,
        "prompt_hash_sha256": prompt_hash,
    }


# Backwards compatible wrapper: keep your old function name if other notebooks import it.
def dsbg_build_drift_report_and_pruning_lists(*args, **kwargs):
    return dsbg_build_audit_report_and_pruning_lists(*args, **kwargs)


if __name__ == "__main__":
    JSONL_BASE_DIR = Path(
        "/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store/AISI_demo/stage_3/prod/llm_negative_selection/deepseek/bge_large"
    )
    NPZ_BASE_DIR = Path(
        "/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store/AISI_demo/stage_3/prod/llm_negative_selection/bge_large"
    )

    llm_model = "DeepSeek-R1-Distill-Qwen-32B"
    prompt_text = "PASTE YOUR PROMPT HERE"

    dsbg_build_audit_report_and_pruning_lists(
        jsonl_base_dir=JSONL_BASE_DIR,
        npz_base_dir=NPZ_BASE_DIR,
        llm_model=llm_model,
        prompt_text=prompt_text,
        jsonl_glob="*.jsonl",
        reports_subdir="reports/deepseek_bge_audit",
        trunc_ctx_fields=1600,
        sample_trunc=650,
        max_final_expected=3,
        random_seed=7,
        random_n_any=20,
        random_n_stratum=12,
        select_for_rerun_flags=("it_drift", "managerial_drift", "empty_final", "final_not_subset", "missing_context"),
    )