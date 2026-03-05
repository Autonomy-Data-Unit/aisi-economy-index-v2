# ============================================================
# HELPERS
# ============================================================
def safe_get_full_desc(obj: dict) -> str:
    # try common keys, fall back to empty string
    return (
        obj.get("job_description")
        or obj.get("description")
        or obj.get("full_text")
        or ""
    )

def scan_jsonl_dir(jsonl_dir: Path, job_id_set: set[str]):
    title_map = {}
    sector_map = {}
    desc_map = {}

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
    total_files = len(jsonl_files)

    found = 0
    t0 = time.time()
    last_print = t0

    for fi, fn in enumerate(jsonl_files, 1):
        with fn.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                jid = obj.get("id")
                if jid is None:
                    continue
                jid = str(jid)

                if jid not in job_id_set:
                    continue

                if jid not in title_map:
                    found += 1

                title_map[jid] = obj.get("title") or ""
                sector_map[jid] = obj.get("category_name") or ""
                desc_map[jid] = safe_get_full_desc(obj)

        now = time.time()
        if (now - last_print) >= 2.0 or fi == total_files:
            elapsed = now - t0
            rate = fi / elapsed if elapsed > 0 else 0.0
            eta = (total_files - fi) / rate if rate > 0 else float("inf")
            print(
                f"  [{fi:>3}/{total_files}] {fn.name} | found={found:,}/{len(job_id_set):,} "
                f"| elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
                flush=True,
            )
            last_print = now

        if found >= len(job_id_set):
            print("  early-exit: found all ids", flush=True)
            break

    return title_map, sector_map, desc_map, total_files, found

def require_npz_keys(npz, keys, where):
    miss = set(keys) - set(npz.files)
    if miss:
        raise KeyError(f"{where} missing keys {sorted(miss)} (have {sorted(npz.files)})")
