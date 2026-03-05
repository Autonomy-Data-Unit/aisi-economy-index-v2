# download_model.py
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT = Path("/projects/a5u/adu_dev/aisi-economy-index")
MODEL_ID = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"

# Put mistral snapshot here, separate from HF_HOME / llama cache
MODEL_DIR = PROJECT / "models" / "solidrust" / "Mistral-7B-Instruct-v0.3-AWQ"

# Optional: keep any huggingface temp/cache separate too (still not llama)
HF_HOME = PROJECT / "hf_cache_mistral"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,   # important on shared FS
        cache_dir=str(HF_HOME),         # keeps HF hub cache separate from llama
        resume_download=True,
    )

    print(f"[OK] Downloaded model to: {MODEL_DIR}")
    print(f"[OK] HF cache used:       {HF_HOME}")

if __name__ == "__main__":
    main()
