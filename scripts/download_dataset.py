"""Download the 142 datasets (30 resamples each) into ./data.

Log in once with `hf auth login` (or `huggingface-cli login`) if the dataset
repository is private; a public repository needs no login. Never put a token
in this file.

    python -m scripts.download_dataset
"""

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    # Pull the full repository tree into ./data (automatically resumes interrupted downloads and skips cached files)
    path = snapshot_download(
        repo_id="NeraThudar/NeraThudarDS",
        repo_type="dataset",
        # Materialize files directly under ./data so `src.data_loader.ROOT_DATASET_DIR` finds `<dataset>/<dataset>0_TRAIN.ts`
        local_dir="data",
    )
    print(f"Dataset downloaded to {path}")
