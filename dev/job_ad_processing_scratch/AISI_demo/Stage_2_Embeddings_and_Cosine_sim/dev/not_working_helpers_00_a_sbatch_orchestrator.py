# Helpers
def submit_job(sbatch_file, dry_run=False):
    """Submit sbatch job and return job ID"""
    cmd = ["sbatch", str(sbatch_file)]
    
    if dry_run:
        print(f"[DRY RUN] Would submit: {sbatch_file.name}")
        return f"DRY_{sbatch_file.stem}"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,    check=True, cwd=PROJECT_ROOT)
        # Parse: "Submitted batch job 1234567"
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            job_id = match.group(1)
            print(f"✓ Submitted {sbatch_file.name} → Job ID: {job_id}")
            return job_id
        else:
            print(f"✗ Failed to parse job ID from: {result.stdout}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"✗ Error submitting {sbatch_file.name}:")
        print(f"  stderr: {e.stderr}")
        return None


def get_job_status(job_id):
    """Get status of a job or array job"""
    if job_id.startswith("DRY_"):
        return "DRY_RUN"
    
    cmd = ["squeue", "-j", str(job_id), "-h", "-o", "%T"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            # Job not in queue, check sacct
            cmd_sacct = ["sacct", "-j", str(job_id), "-n", "-o", "State", "-P"]
            result = subprocess.run(cmd_sacct, capture_output=True, text=True)
            if result.stdout.strip():
                states = result.stdout.strip().split('\n')
                # Return most recent state
                return states[0]
            return "UNKNOWN"
        
        statuses = result.stdout.strip().split('\n')
        # For array jobs, could have multiple states
        if len(statuses) > 1:
            # Count states
            from collections import Counter
            state_counts = Counter(statuses)
            return f"{dict(state_counts)}"
        return statuses[0]
    except Exception as e:
        return f"ERROR: {e}"


def wait_for_job(job_id, poll_interval=30, max_wait=7200):
    """Wait for a single job to complete"""
    if job_id.startswith("DRY_"):
        print(f"  [DRY RUN] Skipping wait for {job_id}")
        return "DRY_RUN"
    
    start = time.time()
    while time.time() - start < max_wait:
        status = get_job_status(job_id)
        
        if isinstance(status, str) and status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
            return status
        
        # For array jobs with dict status
        if isinstance(status, str) and status.startswith("{"):
            # Parse dict string
            try:
                status_dict = eval(status)
                if all(s in ["COMPLETED", "FAILED", "CANCELLED"] for s in status_dict.keys()):
                    return status_dict
            except:
                pass
        
        time.sleep(poll_interval)
    
    return "TIMEOUT"


def wait_for_jobs(job_dict, poll_interval=30):
    """Wait for multiple jobs with live status display"""
    if all(jid.startswith("DRY_") for jid in job_dict.values()):
        print("[DRY RUN] All jobs are dry run, skipping wait")
        return {k: "DRY_RUN" for k in job_dict}
    
    results = {}
    print(f"\n{'Model':<10} {'Job ID':<12} {'Status':<30} {'Elapsed':<10}")
    print("-" * 65)
    
    start_time = time.time()
    while len(results) < len(job_dict):
        for model, job_id in job_dict.items():
            if model in results:
                continue
            
            status = get_job_status(job_id)
            elapsed = time.time() - start_time
            elapsed_str = f"{elapsed/60:.1f}m"
            
            # Truncate long status strings
            status_str = str(status)[:28]
            print(f"{model:<10} {job_id:<12} {status_str:<30} {elapsed_str:<10}")
            
            # Check completion
            if isinstance(status, str) and status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "UNKNOWN"]:
                results[model] = status
            elif isinstance(status, str) and status.startswith("{"):
                # Array job with all tasks done
                try:
                    status_dict = eval(status)
                    if all(s in ["COMPLETED", "FAILED", "CANCELLED"] for s in status_dict.keys()):
                        results[model] = status
                except:
                    pass
        
        if len(results) < len(job_dict):
            time.sleep(poll_interval)
            clear_output(wait=True)
            print(f"\n{'Model':<10} {'Job ID':<12} {'Status':<30} {'Elapsed':<10}")
            print("-" * 65)
    
    print(f"\n✓ All jobs completed in {(time.time()-start_time)/60:.1f} minutes")
    return results


def verify_file(filepath, min_size_mb=0.1):
    """Check if file exists and meets size requirement"""
    p = Path(filepath)
    if not p.exists():
        return False, "MISSING"
    
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb < min_size_mb:
        return False, f"TOO_SMALL ({size_mb:.2f}MB)"
    
    return True, f"OK ({size_mb:.1f}MB)"



# ============================================================
# HELPERS
# ============================================================
def load_titles(npz_path: Path):
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    return d["titles"] if "titles" in d.files else None

def pretty_candidates(indices, titles=None):
    if titles is None:
        return [int(i) for i in indices]
    return [str(titles[int(i)]) if 0 <= int(i) < len(titles) else f"<bad_idx:{int(i)}>" for i in indices]

def coerce_job_ids(x):
    try:
        return x.astype(np.int64)
    except Exception:
        return np.array([str(v) for v in x], dtype=object)

def assert_topk(arr, name):
    if arr.ndim != 2:
        raise ValueError(f"{name} expected 2D [n,k], got {arr.shape}")
    if arr.shape[1] != TOPK_EXPECTED:
        raise ValueError(f"{name} expected TOPK={TOPK_EXPECTED}, got k={arr.shape[1]}")

def load_all_parts(root: Path):
    parts = sorted(root.rglob("part_*.npz"))
    if not parts:
        raise FileNotFoundError(f"No part_*.npz under {root}")

    job_ids, rix, rval, tix, tval = [], [], [], [], []
    for p in parts:
        d = np.load(p, allow_pickle=True)
        need = {"job_ids", "role_topk_idx", "role_topk_val", "task_topk_idx", "task_topk_val"}
        miss = need - set(d.files)
        if miss:
            raise KeyError(f"{p} missing keys: {sorted(miss)}")
        job_ids.append(d["job_ids"])
        rix.append(d["role_topk_idx"]); rval.append(d["role_topk_val"])
        tix.append(d["task_topk_idx"]); tval.append(d["task_topk_val"])

    out = {
        "job_ids": np.concatenate(job_ids),
        "role_idx": np.vstack(rix),
        "role_val": np.vstack(rval),
        "task_idx": np.vstack(tix),
        "task_val": np.vstack(tval),
    }
    assert_topk(out["role_idx"], f"{root.name}.role_idx")
    assert_topk(out["task_idx"], f"{root.name}.task_idx")
    return out

def align_k(models, names):
    sets = []
    maps = {}
    for n in names:
        ids = coerce_job_ids(models[n]["job_ids"])
        maps[n] = {jid: i for i, jid in enumerate(ids)}
        sets.append(set(maps[n].keys()))
    common = sorted(set.intersection(*sets))
    if not common:
        raise RuntimeError(f"No overlapping job_ids across {names}")
    idx = {n: np.array([maps[n][j] for j in common], dtype=np.int32) for n in names}
    return common, idx

def jaccard_topk(a_idx, b_idx):
    n = a_idx.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        A = set(map(int, a_idx[i]))
        B = set(map(int, b_idx[i]))
        out[i] = len(A & B) / len(A | B) if (A | B) else 1.0
    return out

def summary(x, name):
    x = np.asarray(x)
    print(f"{name}: mean={x.mean():.3f} p50={np.median(x):.3f} p10={np.quantile(x,0.10):.3f} p90={np.quantile(x,0.90):.3f} min={x.min():.3f} max={x.max():.3f}")

def plot_hist(x, title):
    plt.figure()
    plt.hist(x, bins=np.linspace(0, 1, 11))
    plt.title(title)
    plt.xlabel("Jaccard")
    plt.ylabel("count")
    plt.grid(True)
    plt.show()

def show_examples(nameA, nameB, job_ids, A, B, *, titles=None):
    jac = jaccard_topk(A, B)
    hi = int(np.argmax(jac)); lo = int(np.argmin(jac))

    def dump(i, tag):
        print("\n==============================")
        print(f"{nameA} vs {nameB} | {tag} | job_id={job_ids[i]}")
        print("ROLE A:", pretty_candidates(A[i], titles))
        print("ROLE B:", pretty_candidates(B[i], titles))

    dump(hi, "HIGH overlap")
    dump(lo, "LOW overlap")

def margin_stats(vals, name):
    top1 = vals[:, 0]; top2 = vals[:, 1]
    m = top1 - top2
    print(f"{name} top1: mean={top1.mean():.3f} p50={np.median(top1):.3f} p10={np.quantile(top1,0.10):.3f} p90={np.quantile(top1,0.90):.3f}")
    print(f"{name} margin: mean={m.mean():.3f} p50={np.median(m):.3f} p10={np.quantile(m,0.10):.3f} p90={np.quantile(m,0.90):.3f}")


def top1_metrics(a_idx, b_idx, name):
    a_top1 = a_idx[:, 0]
    b_top1 = b_idx[:, 0]

    exact = np.mean(a_top1 == b_top1)

    a_in_b5 = np.mean([
        a_top1[i] in set(b_idx[i]) for i in range(len(a_top1))
    ])
    b_in_a5 = np.mean([
        b_top1[i] in set(a_idx[i]) for i in range(len(b_top1))
    ])

    print(f"{name}")
    print(f"  top1 == top1:      {exact:.3f}")
    print(f"  A top1 in B top5:  {a_in_b5:.3f}")
    print(f"  B top1 in A top5:  {b_in_a5:.3f}")

print("✓ Helper functions loaded")