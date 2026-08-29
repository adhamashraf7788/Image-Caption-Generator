"""Upload trained model artifacts to a Hugging Face Hub model repo.

Usage:
    python -m scripts.push_to_hub --repo-id your-username/image-caption-generator
    python -m scripts.push_to_hub --repo-id your-username/image-caption-generator --run-dir models/resnet_lstm_regularized --subfolder resnet_lstm_regularized

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def push_to_hub(repo_id: str, run_dir: Path, vocab_path: Path, subfolder: str = "") -> None:
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)

    prefix = f"{subfolder}/" if subfolder else ""
    files_to_upload = {
        run_dir / "best_model.pt": f"{prefix}best_model.pt",
        run_dir / "config.yaml": f"{prefix}config.yaml",
    }
    if not subfolder:  # vocab.json is shared/identical across runs, only upload once at root
        files_to_upload[vocab_path] = "vocab.json"

    for local_path, repo_path in files_to_upload.items():
        if not local_path.exists():
            print(f"WARNING: {local_path} not found, skipping")
            continue
        print(f"Uploading {local_path} -> {repo_id}/{repo_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"\nDone. View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-dir", default="models/base_resnet_lstm")
    parser.add_argument("--vocab-path", default="data/processed/vocab.json")
    parser.add_argument("--subfolder", default="", help="e.g. 'resnet_lstm_regularized' to avoid overwriting baseline")
    args = parser.parse_args()

    push_to_hub(args.repo_id, Path(args.run_dir), Path(args.vocab_path), subfolder=args.subfolder)