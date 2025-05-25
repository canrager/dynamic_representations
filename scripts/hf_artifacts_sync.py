#!/usr/bin/env python3

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, hf_hub_download
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
    
    # 2. Upload local files that don't exist on repo
    if local_artifacts_dir.exists():
        print("\nUploading new local files...")
        try:
            repo_files = set(api.list_repo_files(repo_id))
        except:
            repo_files = set()
        
        for local_file in local_artifacts_dir.rglob("*"):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_artifacts_dir)
                relative_path_str = str(relative_path).replace("\\", "/")  # Normalize for HF
                
                if relative_path_str not in repo_files:
                    print(f"  Uploading: {relative_path_str}")
                    api.upload_file(
                        path_or_fileobj=str(local_file),
                        path_in_repo=relative_path_str,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                else:
                    print(f"  Skipping (exists): {relative_path_str}")
    else:
        print(f"Local artifacts directory doesn't exist: {local_artifacts_dir}")
    
    # 3. Download repo files that don't exist locally
    print("\nDownloading new repo files...")
    try:
        repo_files = api.list_repo_files(repo_id)
        local_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        for repo_file in repo_files:
            local_file_path = local_artifacts_dir / repo_file
            
            if not local_file_path.exists():
                print(f"  Downloading: {repo_file}")
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_file,
                    local_dir=local_artifacts_dir,
                    repo_type="model"
                )
            else:
                print(f"  Skipping (exists): {repo_file}")
    except Exception as e:
        print(f"Error downloading files: {e}")
    
    print("\nSync completed!")

if __name__ == "__main__":
    main()
