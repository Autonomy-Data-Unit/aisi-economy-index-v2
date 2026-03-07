# ---
# jupyter:
#   kernelspec:
#     display_name: .venv
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
async def main(ctx, print) -> {"onet_tables": dict}:
    """Download and extract O*NET 30.0 database."""
    ...

# %% [markdown]
#
# Retrieve input arguments

# %%
from dev_utils import set_node_func_args, show_node_vars
set_node_func_args('fetch_onet')

# %% [markdown]
# # Function body

# %%
#|export
import tempfile
from zipfile import ZipFile

import pandas as pd

from ai_index.const import onet_store_path

extract_dir = onet_store_path / "db_30_0_text"

# Download and extract if not present
if not extract_dir.exists():
    import requests
    url = "https://www.onetcenter.org/dl_files/database/db_30_0_text.zip"
    print(f"fetch_onet: downloading from {url}...")
    onet_store_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        for chunk in resp.iter_content(8192):
            tmp.write(chunk)
    print(f"fetch_onet: extracting to {extract_dir}...")
    with ZipFile(tmp_path) as z:
        z.extractall(onet_store_path)
    import os
    os.remove(tmp_path)
    print(f"fetch_onet: extracted {len(list(extract_dir.glob('*.txt')))} files")

# Read all .txt files into dict of DataFrames
onet_tables = {}
for txt_file in sorted(extract_dir.glob("*.txt")):
    key = txt_file.stem
    onet_tables[key] = pd.read_csv(txt_file, sep="\t", header=0, encoding="utf-8", dtype=str)

onet_tables.pop("Read Me", None)
print(f"fetch_onet: loaded {len(onet_tables)} tables from O*NET 30.0")
for key, df in sorted(onet_tables.items()):
    print(f"  {key}: {df.shape}")

{"onet_tables": onet_tables}  #|func_return_line
