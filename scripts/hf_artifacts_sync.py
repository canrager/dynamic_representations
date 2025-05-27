#!/usr/bin/env python3

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_large_folder, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

def main():
    api = HfApi()
    repo_name = "dyn_rep"
    local_artifacts_dir = Path(__file__).parent.parent / "artifacts"
    
    # Get current user
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{repo_name}"
    
    print(f"Syncing with repository: {repo_id}")
    print(f"Local artifacts directory: {local_artifacts_dir}")
    
    # 1. Create repo if it doesn't exist
    try:
        api.repo_info(repo_id)
        print("Repository already exists")
    except RepositoryNotFoundError:
        print("Creating repository...")
        create_repo(repo_id, repo_type="model")
        print("Repository created")
    
    # 2. Upload local files using upload_large_folder (optimized with resumable uploads)
    if local_artifacts_dir.exists() and any(local_artifacts_dir.rglob("*")):
        print("\nUploading artifacts folder...")
        try:
            upload_large_folder(
                folder_path=str(local_artifacts_dir),
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns=[".cache", ".git", "__pycache__", "*.pyc"]
            )
            print("Upload completed")
        except Exception as e:
            print(f"Upload error: {e}")
    else:
        print(f"Local artifacts directory is empty or doesn't exist: {local_artifacts_dir}")
    
    # 3. Download repo files using snapshot_download (optimized with concurrent downloads)
    print("\nDownloading repo files...")
    try:
        local_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_artifacts_dir),
            local_dir_use_symlinks=False,
            max_workers=8  # Concurrent downloads
        )
        print("Download completed")
    except Exception as e:
        print(f"Download error: {e}")
    
    print("\nSync completed!")

if __name__ == "__main__":
    main()
