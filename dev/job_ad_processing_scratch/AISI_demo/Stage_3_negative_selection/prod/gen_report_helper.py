# -*- coding: utf-8 -*-
"""
One-shot LLM_MODEL report generator:
- Writes FULL_TEXT_REPORT (txt) with header incl. LLM model + prompt
- Writes pooled overlap plots: avg-Jaccard 3x3 heatmap + venn (optional)
- Saves everything under a *notebook-relative* folder:
    ./reports/deep_seek_reports/<timestamp>/
  (i.e., relative to Path.cwd() of the Jupyter kernel)

Notes:
- JSONLs assumed at:
    JSONL_BASE_DIR/LLM_MODEL/<embed>/<month>/*.jsonl
- NPZs assumed at:
    NPZ_BASE_DIR/<embed>/<month>.npz
- Job text keys in NPZ expected (per your snippet):
    job_ad_title, domain, job_sector_category, job_desc, job_tasks, job_description
"""

import json
import re
import statistics
import random
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Optional venn
try:
    from matplotlib_venn import venn3
    HAVE_VENN = True
except Exception:
    HAVE_VENN = False


def generate_LLM_MODEL_full_report_and_plots(
    JSONL_BASE_DIR: Path,
    NPZ_BASE_DIR: Path,
    *,
    llm_model: str,
    prompt_text: str,
    embeds=("bge_large", "e5_large", "gte_large"),
    jsonl_glob="*.jsonl",
    use_latest_per_month=True,
    min_common_all=50,
    sample_trunc_desc=600,
    sample_trunc_tasks=600,
    sample_prob=0.01,
    sample_max=10,
    rng_seed=42,
    reports_subdir="reports/deep_seek_reports",
):
    """
    Returns dict with written paths + metadata.
    """

    # -----------------------------
    # timestamp + output folder
    # -----------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.cwd() / reports_subdir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"{ts}_FULL_TEXT_REPORT.txt"
    heatmap_path = out_dir / f"{ts}_avg_jaccard_heatmap.png"
    venn_path = out_dir / f"{ts}_venn_kept_pairs_pooled.png"

    # -----------------------------
    # helpers
    # -----------------------------
    IT_PATTERNS = [
        r"\bsoftware\b", r"\bdeveloper\b", r"\bengineer\b", r"\bdata\b", r"\bml\b", r"\bai\b",
        r"\bcloud\b", r"\bcyber\b", r"\bsecurity\b", r"\bnetwork\b", r"\bsystems?\b",
        r"\bdatabase\b", r"\bit\b", r"\bdevops\b"
    ]
    _IT_RE = re.compile("|".join(IT_PATTERNS), flags=re.I)
    MONTH_RE = re.compile(r"^adzuna_month\d+$", re.I)

    def _is_it_role(title: str) -> bool:
        if not title:
            return False
        return bool(_IT_RE.search(title))

    def _infer_npz(jsonl_path: Path) -> Path:
        """
        JSONL: .../llm_negative_selection/LLM_MODEL/<embed>/<month>/<file>.jsonl
        NPZ:   .../llm_negative_selection/<embed>/<month>.npz
        """
        month = jsonl_path.parent.name
        embed = jsonl_path.parent.parent.name
        return NPZ_BASE_DIR / embed / f"{month}.npz"

    def _clean_str(x, fallback):
        if x is None:
            return fallback
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return fallback
        return s

    def _truncate(s: str, n: int) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) <= n:
            return s
        return s[:n].rstrip() + " ...[truncated]"

    def _tasks_to_str(t) -> str:
        if t is None:
            return ""
        if isinstance(t, (list, tuple, np.ndarray)):
            return ", ".join([str(x).strip() for x in t if str(x).strip()])
        return str(t).strip()

    def _load_npz_lookup(npz_path: Path):
        """
        job_id -> {title, domain, sector, desc, tasks, full_ad}
        keys expected:
          job_ad_title, domain, job_sector_category, job_desc, job_tasks, job_description
        """
        if not npz_path.exists():
            return {}

        try:
            with np.load(npz_path, allow_pickle=True) as z:
                files = set(z.files)

                job_id_key = next((k for k in ("job_ids", "job_id") if k in files), None)
                if job_id_key is None:
                    return {}

                jids = [str(x).strip() for x in z[job_id_key].tolist()]

                def get_col(key, default=None):
                    if key not in files:
                        return [default] * len(jids)
                    arr = z[key]
                    return arr.tolist() if hasattr(arr, "tolist") else list(arr)

                # IMPORTANT: use the exact keys you validated
                titles = get_col("job_ad_title", default=None)
                domains = get_col("domain", default=None)
                sectors = get_col("job_sector_category", default=None)

                job_descs = get_col("job_desc", default=None)
                job_tasks = get_col("job_tasks", default=None)
                full_ads = get_col("job_description", default=None)

            lookup = {}
            for i, jid in enumerate(jids):
                if not jid or jid.lower() in ("nan", "none"):
                    continue
                lookup[jid] = {
                    "title": _clean_str(titles[i], "N/A"),
                    "domain": _clean_str(domains[i], "UNKNOWN"),
                    "sector": _clean_str(sectors[i], "UNKNOWN"),
                    "desc": _clean_str(job_descs[i], ""),
                    "tasks": _tasks_to_str(job_tasks[i]),
                    "full_ad": _clean_str(full_ads[i], ""),
                }
            return lookup

        except Exception as e:
            print(f"[WARN] Error loading NPZ {npz_path}: {e}")
            return {}

    class ModelStats:
        def __init__(self, name):
            self.name = name
            self.before_counts = []
            self.after_counts = []
            self.empty_outputs = 0
            self.total_kept = 0
            self.it_leak = 0
            self.kept_titles = Counter()
            self.domain_kept_counts = defaultdict(list)
            self.samples = []

        def add_job(self, jid, candidates, final, ctx, dropped=None):
            candidates = candidates or []
            final = final or []
            dropped = dropped or []

            b_len = len(candidates) if candidates else len(final)
            a_len = len(final)

            self.before_counts.append(b_len)
            self.after_counts.append(a_len)

            if a_len == 0:
                self.empty_outputs += 1

            for t in final:
                self.kept_titles[t] += 1
                self.total_kept += 1
                if _is_it_role(t):
                    self.it_leak += 1

            dom = ctx.get("domain", "UNKNOWN")
            self.domain_kept_counts[dom].append(a_len)

            # sample ~10 cases total
            if len(self.samples) < sample_max and random.random() < sample_prob:
                self.samples.append({
                    "job_id": jid,
                    "title": ctx.get("title"),
                    "domain": ctx.get("domain"),
                    "sector": ctx.get("sector"),
                    "desc": ctx.get("desc", ""),
                    "tasks": ctx.get("tasks", ""),
                    "full_ad": ctx.get("full_ad", ""),
                    "kept": final,
                    "drop": dropped,
                })

    # ---- overlap helpers (pooled) ----
    def list_months(base: Path):
        months = set()
        for emb in embeds:
            d = base / emb
            if not d.exists():
                continue
            for m in d.iterdir():
                if m.is_dir() and MONTH_RE.match(m.name):
                    months.add(m.name)
        return sorted(months)

    def pick_jsonls(base: Path, emb: str, month: str):
        d = base / emb / month
        if not d.exists():
            return []
        files = list(d.glob(jsonl_glob))
        if not files:
            files = list(d.glob("*.jsonl"))
        if not files:
            return []
        if use_latest_per_month:
            return [max(files, key=lambda p: p.stat().st_mtime)]
        return sorted(files)

    def load_job_to_keptset_and_pairs(jsonl_paths):
        job_to_set = {}
        pairs = set()  # (job_id, kept_title)
        for p in jsonl_paths:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    jid = str(obj.get("job_id", "")).strip()
                    if not jid:
                        continue
                    kept = obj.get("final", []) or []
                    kept_set = frozenset([str(x).strip() for x in kept if str(x).strip()])
                    job_to_set[jid] = kept_set
                    for t in kept_set:
                        pairs.add((jid, t))
        return job_to_set, pairs

    def jaccard(a: frozenset, b: frozenset):
        if not a and not b:
            return 1.0
        u = len(a | b)
        return (len(a & b) / u) if u else 0.0

    def save_heatmap(mat, labels, title, out_path):
        fig = plt.figure(figsize=(6.4, 5.3))
        ax = fig.add_subplot(111)

        im = ax.imshow(mat, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title)

        for i in range(len(labels)):
            for j in range(len(labels)):
                v = mat[i, j]
                txt = "NA" if np.isnan(v) else f"{v:.3f}"
                ax.text(j, i, txt, ha="center", va="center")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    # -----------------------------
    # RUN: parse jsonls + load npz
    # -----------------------------
    random.seed(rng_seed)

    stats_by_embed = {e: ModelStats(e.upper()) for e in embeds}

    # JSONLs are expected under JSONL_BASE_DIR/<embed>/<month>/*.jsonl
    all_jsonls = []
    for emb in embeds:
        emb_dir = JSONL_BASE_DIR / emb
        if not emb_dir.exists():
            print(f"[WARN] Missing embed dir: {emb_dir}")
            continue
        all_jsonls.extend(list(emb_dir.rglob(jsonl_glob)))
    all_jsonls = sorted(all_jsonls)
    print(all_jsonls)
    if not all_jsonls:
        raise RuntimeError(f"No JSONLs found under {JSONL_BASE_DIR} with glob={jsonl_glob}")

    # cache npz lookup per (embed, month)
    npz_cache = {}  # (embed, month) -> lookup dict

    for fpath in all_jsonls:
        emb = fpath.parent.parent.name
        month = fpath.parent.name

        if emb not in stats_by_embed:
            continue

        stats = stats_by_embed[emb]

        cache_key = (emb, month)
        if cache_key not in npz_cache:
            npz_path = _infer_npz(fpath)
            if not npz_path.exists():
                print(f"[WARN] NPZ missing for {emb}/{month}: {npz_path}")
                npz_cache[cache_key] = {}
            else:
                npz_cache[cache_key] = _load_npz_lookup(npz_path)

        lookup = npz_cache[cache_key]

        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                jid = str(obj.get("job_id", "")).strip()
                if not jid:
                    continue

                ctx = lookup.get(jid, {})
                cand = obj.get("candidates", []) or []
                final = obj.get("final", []) or []
                dropped = obj.get("drop", []) or []
                stats.add_job(jid, cand, final, ctx, dropped=dropped)

    # -----------------------------
    # BUILD TXT REPORT
    # -----------------------------
    lines = []
    lines.append("=" * 90)
    lines.append("LLM_MODEL FULL REPORT + OVERLAP PLOTS")
    lines.append("=" * 90)
    lines.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"llm_model: {llm_model}")
    lines.append(f"cwd: {Path.cwd()}")
    lines.append(f"jsonl_base_dir: {JSONL_BASE_DIR}")
    lines.append(f"npz_base_dir: {NPZ_BASE_DIR}")
    lines.append(f"out_dir: {out_dir}")
    lines.append(f"embeds: {list(embeds)}")
    lines.append(f"jsonl_glob: {jsonl_glob}")
    lines.append(f"use_latest_per_month (plots): {use_latest_per_month}")
    lines.append("")
    lines.append("prompt_used:")
    lines.append("-" * 90)
    lines.append(prompt_text.strip())
    lines.append("-" * 90)
    lines.append("")

    for emb in embeds:
        stats = stats_by_embed[emb]
        if not stats.before_counts:
            lines.append("=" * 70)
            lines.append(f"{emb.upper()}")
            lines.append("=" * 70)
            lines.append("NO DATA (no JSONLs parsed)")
            lines.append("")
            continue

        avg_before = statistics.mean(stats.before_counts)
        avg_after = statistics.mean(stats.after_counts)
        drop_rate = 1.0 - (avg_after / avg_before) if avg_before > 0 else 0.0
        it_share = stats.it_leak / max(stats.total_kept, 1)

        lines.append("=" * 70)
        lines.append(f"{stats.name}")
        lines.append("=" * 70)
        lines.append("GLOBAL METRICS")
        lines.append(f"jobs: {len(stats.before_counts)}")
        lines.append(f"avg_candidates_before: {avg_before:.3f}")
        lines.append(f"avg_candidates_after:  {avg_after:.3f}")
        lines.append(f"drop_rate:            {drop_rate:.4f}")
        lines.append(f"empty_outputs_percent:{(100 * stats.empty_outputs / len(stats.before_counts)):.2f}")
        lines.append(f"it_leakage_share:     {it_share:.4f}")
        lines.append(f"min_kept:             {min(stats.after_counts)}")
        lines.append(f"max_kept:             {max(stats.after_counts)}")
        lines.append("")

        lines.append("TOP KEPT OCCUPATIONS (top 25)")
        for title, count in stats.kept_titles.most_common(25):
            lines.append(f"{count:>6}  {title}")
        lines.append("")

        lines.append("DOMAIN AVG KEPT")
        domain_avgs = {k: statistics.mean(v) for k, v in stats.domain_kept_counts.items()}
        for dom, val in sorted(domain_avgs.items(), key=lambda x: (-x[1], x[0])):
            # keep formatting stable
            val_str = f"{val:.3f}".rstrip("0").rstrip(".")
            lines.append(f"{dom}: {val_str}")
        lines.append("")

        lines.append("SAMPLE CASES (truncated)")
        for s in stats.samples:
            lines.append("")
            lines.append(f"job_id: {s['job_id']}")
            lines.append(f"title:  {s.get('title')}")
            lines.append(f"domain: {s.get('domain')} | sector: {s.get('sector')}")
            if s.get("desc"):
                lines.append(f"desc:   {_truncate(s['desc'], sample_trunc_desc)}")
            if s.get("tasks"):
                lines.append(f"tasks:  {_truncate(s['tasks'], sample_trunc_tasks)}")
            # optional: include full_ad snippet if you want
            # if s.get("full_ad"):
            #     lines.append(f"full_ad:{_truncate(s['full_ad'], 700)}")
            lines.append(f"kept:   {s.get('kept')}")
            lines.append(f"drop:   {s.get('drop')}")
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # -----------------------------
    # POOLED OVERLAP PLOTS
    # -----------------------------
    months = list_months(JSONL_BASE_DIR)
    pooled_jobsets = {emb: {} for emb in embeds}   # month::job_id -> kept_set
    pooled_pairs = {emb: set() for emb in embeds}  # (month::job_id, kept_title)
    loaded_months = 0

    for month in months:
        per_jobsets = {}
        per_pairs = {}

        ok = True
        for emb in embeds:
            jsonls = pick_jsonls(JSONL_BASE_DIR, emb, month)
            if not jsonls:
                ok = False
                break
            job_to_set, pairs = load_job_to_keptset_and_pairs(jsonls)
            per_jobsets[emb] = job_to_set
            per_pairs[emb] = pairs

        if not ok:
            continue

        # intersection on jobs present in all 3 embeds
        common_all = set(per_jobsets[embeds[0]])
        for emb in embeds[1:]:
            common_all &= set(per_jobsets[emb])

        if len(common_all) < min_common_all:
            continue

        # pool only the common jobs (namespace by month)
        for emb in embeds:
            for jid in common_all:
                pooled_jobsets[emb][f"{month}::{jid}"] = per_jobsets[emb][jid]
            pooled_pairs[emb] |= set(
                (f"{month}::{jid}", t) for (jid, t) in per_pairs[emb] if jid in common_all
            )

        loaded_months += 1

    # avg-jaccard matrix
    labels = [e.upper() for e in embeds]
    mat = np.full((len(embeds), len(embeds)), np.nan, dtype=float)

    for i, a in enumerate(embeds):
        for j, b in enumerate(embeds):
            if i == j:
                mat[i, j] = 1.0
                continue
            common = sorted(set(pooled_jobsets[a]) & set(pooled_jobsets[b]))
            if not common:
                continue
            vals = [jaccard(pooled_jobsets[a][jid], pooled_jobsets[b][jid]) for jid in common]
            mat[i, j] = float(np.mean(vals))

    save_heatmap(
        mat,
        labels,
        title=f"Pooled avg Jaccard of kept sets (n_months={loaded_months}, min_common={min_common_all})",
        out_path=heatmap_path
    )

    # venn on (month::job_id, kept_title)
    venn_written = None
    if HAVE_VENN:
        A = pooled_pairs[embeds[0]]
        B = pooled_pairs[embeds[1]]
        C = pooled_pairs[embeds[2]]

        fig = plt.figure(figsize=(7.2, 5.2))
        ax = fig.add_subplot(111)
        venn3([A, B, C], set_labels=(labels[0], labels[1], labels[2]), ax=ax)
        ax.set_title(f"Pooled overlap of kept labels (job_id,title) (n_months={loaded_months})")
        fig.tight_layout()
        fig.savefig(venn_path, dpi=170)
        plt.close(fig)
        venn_written = venn_path
    else:
        # don't leave a path that doesn't exist
        venn_path = None

    print("[OK] Wrote:")
    print(f"  txt:     {txt_path}")
    print(f"  heatmap: {heatmap_path}")
    if venn_written:
        print(f"  venn:    {venn_written}")
    else:
        print("  venn:    skipped (install matplotlib-venn)")

    return {
        "txt_path": str(txt_path),
        "heatmap_path": str(heatmap_path),
        "venn_path": (str(venn_written) if venn_written else None),
        "out_dir": str(out_dir),
        "timestamp": ts,
        "loaded_months_for_plots": loaded_months,
        "cwd": str(Path.cwd()),
    }
