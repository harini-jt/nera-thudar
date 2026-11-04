from huggingface_hub import snapshot_download
from huggingface_hub import login
from huggingface_hub import whoami

login()
user = whoami(token="PUT TOKEN_HERE")
print(f"Logged in as: {user}")
snapshot_download(
    repo_id="NeraThudar/NeraThudarDS",
    repo_type="dataset",
    local_dir="data",
    token="PUT TOKEN_HERE"
)