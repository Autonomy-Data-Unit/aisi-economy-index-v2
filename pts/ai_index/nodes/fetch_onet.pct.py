# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fetch O*NET
#
# Load O*NET 30.0 database from the local store (or download if missing).
# Returns all tables as a dict of DataFrames.

# %%
#|default_exp nodes.fetch_onet
#|export_as_func true

# %%
#|set_func_signature
def fetch_onet(ctx, print) -> {"onet_tables": dict}:
    """Download and extract O*NET 30.0 database."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dotenv import load_dotenv; load_dotenv()
from dev_utils import set_node_func_args
set_node_func_args(fetch_onet)

# %%
#|export
from zipfile import ZipFile

import pandas as pd

from ai_index.const import onet_store_path

zip_path = onet_store_path / "db_30_0_text.zip"

# Download if not present
if not zip_path.exists():
    import requests
    url = "https://www.onetcenter.org/dl_files/database/db_30_0_text.zip"
    print(f"fetch_onet: downloading from {url}...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as fh:
        for chunk in resp.iter_content(8192):
            fh.write(chunk)
    print(f"fetch_onet: downloaded {zip_path.stat().st_size / 1e6:.1f} MB")

# Read all .txt files into dict of DataFrames
onet_tables = {}
with ZipFile(zip_path) as z:
    for info in z.infolist():
        if not info.filename.endswith(".txt"):
            continue
        key = info.filename.replace("db_30_0_text/", "").replace(".txt", "")
        onet_tables[key] = pd.read_csv(
            z.open(info.filename), sep="\t", header=0, encoding="utf-8", dtype=str
        )

onet_tables.pop("Read Me", None)
print(f"fetch_onet: loaded {len(onet_tables)} tables from O*NET 30.0")
for key, df in sorted(onet_tables.items()):
    print(f"  {key}: {df.shape}")

{"onet_tables": onet_tables}  #|func_return_line
