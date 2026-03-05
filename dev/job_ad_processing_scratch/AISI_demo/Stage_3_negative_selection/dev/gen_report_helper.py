# -*- coding: utf-8 -*-
"""
One-shot LLM_MODEL report generator (Comprehensive Version):
- Scans strictly Month 1 to Month 14.
- Handles flexible directory names (adzuna_month1 vs adzuna_month01).
- Generates a FULL detailed text report (Global Metrics, Top Occupations, Domain Stats, Samples).
- Generates pooled overlap plots (Heatmap + Venn).
- Prints a terminal audit trail and availability matrix.
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
    jobid_by_embed=None,  
):
    # -----------------------------
    # Setup: Timestamp + Output Folder
    # -----------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.cwd() / reports_subdir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / f"{ts}_FULL_TEXT_REPORT.txt"
    heatmap_path = out_dir / f"{ts}_avg_jaccard_heatmap.png"
    venn_path = out_dir / f"{ts}_venn_kept_pairs_pooled.png"

    # Audit matrix to track which files were found
    availability_matrix = {emb: {} for emb in embeds}
    
    # -----------------------------
    # Helpers
    # -----------------------------
    IT_PATTERNS = [
        r"\bsoftware\b", r"\bdeveloper\b", r"\bengineer\b", r"\bdata\b", r"\bml\b", r"\bai\b",
        r"\bcloud\b", r"\bcyber\b", r"\bsecurity\b", r"\bnetwork\b", r"\bsystems?\b",
        r"\bdatabase\b", r"\bit\b", r"\bdevops\b"
    ]
    _IT_RE = re.compile("|".join(IT_PATTERNS), flags=re.I)

    def _is_it_role(title: str) -> bool:
        return bool(_IT_RE.search(title)) if title else False

    def _infer_npz(jsonl_path: Path) -> Path:
        month = jsonl_path.parent.name
        embed = jsonl_path.parent.parent.name
        return NPZ_BASE_DIR / embed / f"{month}.npz"

    def _clean_str(x, fallback):
        if x is None: return fallback
        s = str(x).strip()
        return fallback if s == "" or s.lower() in ("nan", "none", "null") else s

    def _truncate(s: str, n: int) -> str:
        if not s: return ""
        s = str(s)
        return s if len(s) <= n else s[:n].rstrip() + " ...[truncated]"

    def _tasks_to_str(t) -> str:
        if t is None: return ""
        if isinstance(t, (list, tuple, np.ndarray)):
            return ", ".join([str(x).strip() for x in t if str(x).strip()])
        return str(t).strip()

    def _load_npz_lookup(npz_path: Path):
        """Loads NPZ data to enrich the report with Domain/Sector/Description info."""
        if not npz_path.exists(): return {}
        try:
            with np.load(npz_path, allow_pickle=True) as z:
                files = set(z.files)
                job_id_key = next((k for k in ("job_ids", "job_id") if k in files), None)
                if job_id_key is None: return {}
                jids = [str(x).strip() for x in z[job_id_key].tolist()]
                
                # Helper to safely get columns
                def get_col(key, default=None):
                    if key not in files: return [default] * len(jids)
                    arr = z[key]
                    return arr.tolist() if hasattr(arr, "tolist") else list(arr)

                titles = get_col("job_ad_title")
                domains = get_col("domain")
                sectors = get_col("job_sector_category")
                descs = get_col("job_desc")
                tasks = get_col("job_tasks")
                full_ads = get_col("job_description")

                return {
                    jid: {
                        "title": _clean_str(titles[i], "N/A"),
                        "domain": _clean_str(domains[i], "UNKNOWN"),
                        "sector": _clean_str(sectors[i], "UNKNOWN"),
                        "desc": _clean_str(descs[i], ""),
                        "tasks": _tasks_to_str(tasks[i]),
                        "full_ad": _clean_str(full_ads[i], ""),
                    }
                    for i, jid in enumerate(jids) 
                    if jid and jid.lower() not in ("nan", "none")
                }
        except Exception as e:
            print(f"[WARN] Error loading NPZ {npz_path}: {e}")
            return {}

    class ModelStats:
        """Tracks detailed stats per embedding model."""
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

            if len(self.samples) < sample_max and random.random() < sample_prob:
                self.samples.append({
                    "job_id": jid,
                    "title": ctx.get("title"),
                    "domain": dom,
                    "sector": ctx.get("sector"),
                    "desc": ctx.get("desc", ""),
                    "tasks": ctx.get("tasks", ""),
                    "full_ad": ctx.get("full_ad", ""),
                    "kept": final,
                    "drop": dropped,
                })

    def pick_jsonls(base: Path, emb: str, month_idx: int):
        """
        Smart picker: checks 'adzuna_monthX' AND 'adzuna_month0X'.
        Selects based on jobID (if provided) or timestamp.
        """
        possible_names = [f"adzuna_month{month_idx}", f"adzuna_month{month_idx:02d}"]
        
        for m_name in possible_names:
            d = base / emb / m_name
            if d.exists():
                files = list(d.glob(jsonl_glob)) or list(d.glob("*.jsonl"))
                if files:
                    chosen = None
                    # Strategy 1: Specific Job ID
                    if jobid_by_embed and emb in jobid_by_embed and jobid_by_embed[emb]:
                        jid = str(jobid_by_embed[emb]).strip()
                        matches = [p for p in files if f"_job{jid}_" in p.name]
                        if matches: 
                            chosen = max(matches, key=lambda p: p.stat().st_mtime)
                    
                    # Strategy 2: Latest Timestamp
                    if not chosen and use_latest_per_month:
                        chosen = max(files, key=lambda p: p.stat().st_mtime)
                    
                    if chosen:
                        availability_matrix[emb][month_idx] = "OK"
                        print(f"[SELECT] {emb} | {m_name} -> {chosen.name}")
                        return [chosen]
        
        # If loop finishes without returning
        availability_matrix[emb][month_idx] = "--"
        return []

    # -----------------------------
    # PHASE 1: SCAN FILES (M1 - M14)
    # -----------------------------
    random.seed(rng_seed)
    stats_by_embed = {e: ModelStats(e.upper()) for e in embeds}
    all_jsonls = []

    print(f"\nScanning Month 1 to 14 (Flexible format) for embeds: {embeds}")
    for emb in embeds:
        for i in range(1, 15):
            found = pick_jsonls(JSONL_BASE_DIR, emb, i)
            all_jsonls.extend(found)

    if not all_jsonls:
        raise RuntimeError(f"No JSONLs found for Month 1-14 in {JSONL_BASE_DIR}")

    # -----------------------------
    # PHASE 2: LOAD DATA & STATS
    # -----------------------------
    npz_cache = {}
    print("\nLoading data and calculating stats...")
    
    for fpath in all_jsonls:
        emb = fpath.parent.parent.name
        month = fpath.parent.name
        
        if emb not in stats_by_embed: continue
        
        # Load NPZ context if not cached
        if (emb, month) not in npz_cache:
            npz_cache[(emb, month)] = _load_npz_lookup(_infer_npz(fpath))

        lookup = npz_cache[(emb, month)]
        
        # Stream JSONL lines
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    jid = str(obj.get("job_id", "")).strip()
                    if not jid: continue
                    
                    # Context for this job
                    ctx = lookup.get(jid, {})
                    stats_by_embed[emb].add_job(
                        jid, 
                        obj.get("candidates"), 
                        obj.get("final"), 
                        ctx, 
                        dropped=obj.get("drop")
                    )
                except: continue

    # -----------------------------
    # PHASE 3: WRITE COMPREHENSIVE REPORT
    # -----------------------------
    lines = []
    lines.append("=" * 90)
    lines.append("LLM_MODEL FULL COMPREHENSIVE REPORT")
    lines.append("=" * 90)
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Model: {llm_model}")
    lines.append(f"Output Directory: {out_dir}")
    lines.append("")
    lines.append("PROMPT USED:")
    lines.append("-" * 90)
    lines.append(prompt_text.strip())
    lines.append("-" * 90)
    lines.append("")

    for emb in embeds:
        s = stats_by_embed[emb]
        if not s.before_counts:
            lines.append(f"EMBEDDING: {emb.upper()} - NO DATA FOUND\n")
            continue
        
        # Metrics Calculation
        avg_before = statistics.mean(s.before_counts)
        avg_after = statistics.mean(s.after_counts)
        drop_rate = 1.0 - (avg_after / avg_before) if avg_before > 0 else 0.0
        it_share = s.it_leak / max(s.total_kept, 1)
        empty_pct = (s.empty_outputs / len(s.before_counts)) * 100

        lines.append("=" * 70)
        lines.append(f"EMBEDDING: {s.name}")
        lines.append("=" * 70)
        lines.append("GLOBAL METRICS")
        lines.append(f"  Total Jobs Processed:    {len(s.before_counts)}")
        lines.append(f"  Avg Candidates (Before): {avg_before:.3f}")
        lines.append(f"  Avg Candidates (After):  {avg_after:.3f}")
        lines.append(f"  Drop Rate:               {drop_rate:.4f}")
        lines.append(f"  Empty Outputs:           {s.empty_outputs} ({empty_pct:.2f}%)")
        lines.append(f"  IT Leakage Share:        {it_share:.4f} ({s.it_leak}/{s.total_kept})")
        lines.append("")

        lines.append("TOP KEPT OCCUPATIONS (Top 25)")
        for title, count in s.kept_titles.most_common(25):
            lines.append(f"  {count:>5} x {title}")
        lines.append("")

        lines.append("DOMAIN AVERAGES (Avg items kept per domain)")
        domain_avgs = {k: statistics.mean(v) for k, v in s.domain_kept_counts.items()}
        for dom, val in sorted(domain_avgs.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  {dom:<20}: {val:.3f}")
        lines.append("")

        lines.append("SAMPLE CASES (Truncated)")
        for sample in s.samples:
            lines.append("-" * 40)
            lines.append(f"Job ID: {sample['job_id']}")
            lines.append(f"Title:  {sample.get('title')}")
            lines.append(f"Domain: {sample.get('domain')} | Sector: {sample.get('sector')}")
            if sample.get('desc'):
                lines.append(f"Desc:   {_truncate(sample['desc'], sample_trunc_desc)}")
            if sample.get('tasks'):
                lines.append(f"Tasks:  {_truncate(sample['tasks'], sample_trunc_tasks)}")
            lines.append(f"Kept:   {sample.get('kept')}")
            lines.append(f"Dropped:{sample.get('drop')}")
        lines.append("\n")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # -----------------------------
    # PHASE 4: GENERATE PLOTS (POOLED)
    # -----------------------------
    # Logic: Group by (Month + JobID) to ensure unique keys across time
    pooled_jobsets = {emb: {} for emb in embeds}
    pooled_pairs = {emb: set() for emb in embeds}
    loaded_months = 0

    for i in range(1, 15):
        # We check both month folder names again to match what we loaded
        # Filter all_jsonls for this specific month index
        month_patterns = (f"adzuna_month{i}", f"adzuna_month{i:02d}")
        
        per_jobsets = {} # emb -> {jid: set}
        per_pairs = {}   # emb -> set((jid, title))
        
        ok = True
        for emb in embeds:
            # Find the file for this month/embed in our pre-scanned list
            relevant_files = [
                p for p in all_jsonls 
                if p.parent.name in month_patterns and p.parent.parent.name == emb
            ]
            
            if not relevant_files:
                ok = False
                break
            
            # Load stats for overlap (using the first found file, there should only be 1 per month per embed)
            j_to_s = {}
            p_s = set()
            with relevant_files[0].open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        jid = str(obj.get("job_id")).strip()
                        kept = frozenset([str(x).strip() for x in (obj.get("final") or []) if str(x).strip()])
                        j_to_s[jid] = kept
                        for t in kept: p_s.add((jid, t))
                    except: continue
            
            per_jobsets[emb] = j_to_s
            per_pairs[emb] = p_s

        if ok:
            # Intersection of Job IDs across all 3 models
            common_jids = set(per_jobsets[embeds[0]])
            for e in embeds[1:]:
                common_jids &= set(per_jobsets[e])
            
            if len(common_jids) >= min_common_all:
                for emb in embeds:
                    for jid in common_jids:
                        # Key must include month to differentiate same job ID appearing in different months (unlikely but safe)
                        key = f"m{i}::{jid}"
                        pooled_jobsets[emb][key] = per_jobsets[emb][jid]
                    
                    # Pairs are just added to the big pool
                    pooled_pairs[emb] |= {(f"m{i}::{jid}", t) for (jid, t) in per_pairs[emb] if jid in common_jids}
                
                loaded_months += 1

    # --- Heatmap ---
    labels = [e.upper() for e in embeds]
    mat = np.eye(len(embeds))
    
    def jaccard(s1, s2):
        u = len(s1 | s2)
        return len(s1 & s2) / u if u else 1.0

    for i, a in enumerate(embeds):
        for j, b in enumerate(embeds):
            if i >= j: continue
            # Common keys in the POOLED dictionary
            common_keys = set(pooled_jobsets[a]) & set(pooled_jobsets[b])
            if common_keys:
                vals = [jaccard(pooled_jobsets[a][k], pooled_jobsets[b][k]) for k in common_keys]
                mat[i,j] = mat[j,i] = np.mean(vals)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for (ii,jj), v in np.ndenumerate(mat): ax.text(jj,ii,f"{v:.3f}", ha="center", va="center", color="white" if v < 0.5 else "black")
    plt.title(f"Avg Jaccard (Pooled M1-14, n_months={loaded_months})")
    plt.tight_layout(); plt.savefig(heatmap_path, dpi=150); plt.close()

    # --- Venn ---
    if HAVE_VENN:
        fig, ax = plt.subplots(figsize=(7, 5))
        venn3([pooled_pairs[e] for e in embeds], set_labels=labels, ax=ax)
        plt.title(f"Label Overlap (Pooled M1-14)")
        plt.tight_layout(); plt.savefig(venn_path, dpi=150); plt.close()

    # -----------------------------
    # PHASE 5: TERMINAL OUTPUT
    # -----------------------------
    print("\n" + "="*40)
    print(" DATA AVAILABILITY MATRIX (M1 - M14)")
    print("="*40)
    header = "Month | " + " | ".join([f"{e[:8]:8}" for e in embeds])
    print(header)
    print("-" * len(header))
    for i in range(1, 15):
        row = f"{i:5} | " + " | ".join([f"{availability_matrix[e].get(i, '--'):8}" for e in embeds])
        print(row)
    print("="*40)
    
    print(f"\n[OK] Processing Complete.")
    print(f"Report saved to: {txt_path}")
    print(f"Heatmap saved to: {heatmap_path}")
    
    return {
        "out_dir": str(out_dir),
        "txt_path": str(txt_path),
        "heatmap_path": str(heatmap_path)
    }