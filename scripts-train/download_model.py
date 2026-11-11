from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    local_dir=r"E:\Hosted LLM\models\smollm2-135m"
)

print("Model downloaded successfully!")