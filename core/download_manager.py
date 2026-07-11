import os
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_CONFIGS = {
    "local": {
        "type": "snapshot",
        "repo_id": "intfloat/multilingual-e5-small",
        "local_dir": os.path.join(MODELS_DIR, "multilingual-e5-small")
    },
    "gemma": {
        "type": "snapshot",
        "repo_id": "google/gemma-4-E4B",
        "local_dir": os.path.join(MODELS_DIR, "gemma_4_e4b")
    },
    "mistral": {
        "type": "file",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "file_name": "mistral-7b-instruct-v0.1.Q4_0.gguf",
        "local_dir": os.path.join(MODELS_DIR, "mistral_7b_instruct_v0.1"),
        "local_path": os.path.join(MODELS_DIR, "mistral_7b_instruct_v0.1", "mistral-7b-v0.1.Q4_0.gguf")
    }
}


class DownloadManager:
    @staticmethod
    def check_model_status() -> dict:
        status = {}
        for model_id, config in MODEL_CONFIGS.items():
            if config["type"] == "snapshot":
                status[model_id] = os.path.exists(config["local_dir"]) and len(os.listdir(config["local_dir"])) > 0
            else:
                status[model_id] = os.path.exists(config["local_path"])
        return status

    @classmethod
    def download_model_sync(cls, model_id: str):
        if model_id not in MODEL_CONFIGS:
            raise ValueError("مدل نامعتبر است")

        config = MODEL_CONFIGS[model_id]

        if config["type"] == "snapshot":
            print(f"[HF HUB] Starting snapshot download for: {config['repo_id']}")
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=config["local_dir"],
                local_dir_use_symlinks=False
            )
            print(f"[HF HUB] Snapshot download completed for {model_id}.")
        else:
            print(f"[HF HUB] Starting file download: {config['file_name']} from {config['repo_id']}")
            downloaded_file = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["file_name"],
                local_dir=config["local_dir"],
                local_dir_use_symlinks=False
            )

            if downloaded_file != config["local_path"]:
                os.makedirs(os.path.dirname(config["local_path"]), exist_ok=True)
                if os.path.exists(config["local_path"]):
                    os.remove(config["local_path"])
                os.rename(downloaded_file, config["local_path"])

            print(f"[HF HUB] File download completed for {model_id}.")