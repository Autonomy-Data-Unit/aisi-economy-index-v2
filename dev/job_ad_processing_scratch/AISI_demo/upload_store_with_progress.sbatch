#!/bin/bash
#SBATCH --job-name=z_upload_store
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00

set -euo pipefail

# ---- activate python env ----
source /projects/a5u/adu_dev/aisi-economy-index/.venv/bin/activate

# ---- paths / s3 ----
TAR="/lus/lfs1aip2/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store_20260227.tar.gz"
BUCKET="adu-project-data"
KEY="aisi-economy-index/pre_migration_backup/store_20260227.tar.gz"
PROFILE=""   # leave empty unless you use a named AWS profile

echo "Start UTC: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
echo "File:"
ls -lh "$TAR"

python -u - <<'PY'
import time
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig

tar_path = Path("/lus/lfs1aip2/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store_20260227.tar.gz")
bucket = "adu-project-data"
key = "aisi-economy-index/pre_migration_backup/store_20260227.tar.gz"
profile = None

session = boto3.Session(profile_name=profile) if profile else boto3.Session()
s3 = session.client("s3")

total = tar_path.stat().st_size

class Progress:
    def __init__(self, total_bytes, step_gib=10):
        self.total = total_bytes
        self.step = int(step_gib * 1024**3)
        self.seen = 0
        self.next_mark = self.step
        self.t0 = time.time()

    def __call__(self, bytes_amount):
        self.seen += bytes_amount
        if self.seen >= self.next_mark:
            elapsed = time.time() - self.t0
            gib = self.seen / 1024**3
            rate = gib / elapsed if elapsed > 0 else 0
            pct = 100 * self.seen / self.total
            print(f"Uploaded {gib:.1f} GiB ({pct:.1f}%), avg {rate:.2f} GiB/s", flush=True)
            while self.next_mark <= self.seen:
                self.next_mark += self.step

print(f"Starting upload of {total/1024**3:.2f} GiB -> s3://{bucket}/{key}", flush=True)

config = TransferConfig(
    multipart_threshold=64 * 1024**2,
    multipart_chunksize=64 * 1024**2,
    max_concurrency=8,
    use_threads=True,
)

s3.upload_file(str(tar_path), bucket, key, Config=config, Callback=Progress(total, step_gib=10))
print("Upload complete", flush=True)
PY

echo "End UTC: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"