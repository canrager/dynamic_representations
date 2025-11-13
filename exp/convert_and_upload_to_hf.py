"""
Convert temporal SAE checkpoints from .pt to .safetensors and upload to Hugging Face.
Handles nested directory structure: model/layer/sae_type/
"""

import torch as th
from safetensors.torch import save_file
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import os
import argparse
import yaml


def convert_pt_to_safetensors(pt_path, safetensors_path, config_path):
    """Convert PyTorch checkpoint to safetensors format."""
    print(f"  Loading checkpoint from {pt_path.name}...")
    checkpoint = th.load(pt_path, map_location="cpu", weights_only=False)

    # Read dtype from config
    target_dtype = th.float32  # default
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            dtype_str = config.get('data', {}).get('dtype', 'float32')
            dtype_map = {
                'float32': th.float32,
                'float16': th.float16,
                'bfloat16': th.bfloat16,
            }
            target_dtype = dtype_map.get(dtype_str, th.float32)
            print(f"  Using dtype: {dtype_str}")

    # Extract the SAE state dict
    if "sae" in checkpoint:
        state_dict = checkpoint["sae"]
    else:
        state_dict = checkpoint

    # # Ensure W_enc is present (handle different naming conventions)
    # if "E" not in state_dict and "D" in state_dict:
    #     # If weights are shared, W_enc = W_dec.T
    #     print(f"  Adding E as D.T...")
    #     state_dict["E"] = state_dict["D"].T.contiguous().clone()


    # Convert to target dtype and make all tensors contiguous
    state_dict = {
        k: v.squeeze().to(target_dtype).contiguous() 
        if isinstance(v, th.Tensor) else v
        for k, v in state_dict.items()
    }

    print(f"  Saving to safetensors format...")
    save_file(state_dict, safetensors_path)
    print(f"  ✓ Converted: {pt_path.relative_to(pt_path.parent.parent.parent.parent)}")

    return state_dict


def find_all_checkpoints(base_dir, temporal_only=False):
    """Find all checkpoint directories containing .pt and .yaml files.

    Args:
        base_dir: Base directory to search
        temporal_only: If True, only include temporal SAE directories
    """
    base_path = Path(base_dir)
    checkpoint_dirs = []

    for pt_file in base_path.rglob("latest_ckpt.pt"):
        checkpoint_dir = pt_file.parent
        yaml_file = checkpoint_dir / "conf.yaml"

        if yaml_file.exists():
            # Filter for temporal SAEs if requested
            if temporal_only and "temporal" not in checkpoint_dir.name.lower():
                continue
            checkpoint_dirs.append(checkpoint_dir)

    return checkpoint_dirs


def upload_to_huggingface(base_dir, repo_id, token=None, temporal_only=False):
    """Upload all checkpoints to Hugging Face maintaining directory structure."""
    print(f"\nUploading to Hugging Face repo: {repo_id}")
    base_path = Path(base_dir)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, exist_ok=True)
        print(f"Repository {repo_id} ready\n")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Upload files
    api = HfApi()

    # Find all checkpoint directories
    checkpoint_dirs = find_all_checkpoints(base_dir, temporal_only=temporal_only)

    for checkpoint_dir in checkpoint_dirs:
        # Get relative path from base directory
        rel_path = checkpoint_dir.relative_to(base_path)

        print(f"Uploading {rel_path}/...")

        # Upload safetensors file
        safetensors_file = checkpoint_dir / "latest_ckpt.safetensors"
        if safetensors_file.exists():
            repo_path = str(rel_path / "latest_ckpt.safetensors")
            api.upload_file(
                path_or_fileobj=str(safetensors_file),
                path_in_repo=repo_path,
                repo_id=repo_id,
                token=token,
            )
            print(f"  ✓ Uploaded: {repo_path}")

        # Upload config file
        config_file = checkpoint_dir / "conf.yaml"
        if config_file.exists():
            repo_path = str(rel_path / "conf.yaml")
            api.upload_file(
                path_or_fileobj=str(config_file),
                path_in_repo=repo_path,
                repo_id=repo_id,
                token=token,
            )
            print(f"  ✓ Uploaded: {repo_path}")

    print(f"\n{'='*60}")
    print(f"Upload complete! View at: https://huggingface.co/{repo_id}")
    print(f"{'='*60}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert and upload SAE checkpoints to Hugging Face"
    )
    parser.add_argument(
        "--temporal-only",
        action="store_true",
        help="Only process temporal SAE directories",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip conversion and only upload existing safetensors files",
    )
    args = parser.parse_args()

    # Base directory containing all checkpoints
    base_dir = Path("./artifacts/trained_saes")

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist!")
        return

    # Find all checkpoint directories
    checkpoint_dirs = find_all_checkpoints(base_dir, temporal_only=args.temporal_only)

    if not checkpoint_dirs:
        filter_msg = " (temporal only)" if args.temporal_only else ""
        print(f"No checkpoints found in {base_dir}{filter_msg}")
        return

    filter_info = " (temporal only)" if args.temporal_only else ""

    if args.upload_only:
        print(f"Found {len(checkpoint_dirs)} checkpoint(s) to upload{filter_info}:\n")
        for ckpt_dir in checkpoint_dirs:
            rel_path = ckpt_dir.relative_to(base_dir)
            print(f"  - {rel_path}")
    else:
        print(f"Found {len(checkpoint_dirs)} checkpoint(s) to convert{filter_info}:\n")
        for ckpt_dir in checkpoint_dirs:
            rel_path = ckpt_dir.relative_to(base_dir)
            print(f"  - {rel_path}")

        print(f"\n{'='*60}")
        print("Converting all checkpoints to safetensors format...")
        print(f"{'='*60}\n")

        # Convert all checkpoints
        for checkpoint_dir in checkpoint_dirs:
            pt_path = checkpoint_dir / "latest_ckpt.pt"
            safetensors_path = checkpoint_dir / "latest_ckpt.safetensors"
            config_path = checkpoint_dir / "conf.yaml"

            try:
                convert_pt_to_safetensors(pt_path, safetensors_path, config_path)
            except Exception as e:
                print(f"  ✗ Error converting {checkpoint_dir.relative_to(base_dir)}: {e}")

    # Upload to HuggingFace
    repo_id = "canrager/temporalSAEs"

    print(f"\n{'='*60}")
    print("IMPORTANT: You need to login to Hugging Face first!")
    print("Run: huggingface-cli login")
    print("Or set HF_TOKEN environment variable")
    print(f"{'='*60}")

    upload_choice = input("\nProceed with upload? (y/n): ")
    if upload_choice.lower() == "y":
        upload_to_huggingface(base_dir, repo_id, temporal_only=args.temporal_only)
    else:
        if args.upload_only:
            print("\nUpload cancelled.")
        else:
            print("\nSkipping upload. You can upload manually later.")
            print(f"Converted files are ready at: {base_dir.absolute()}")


if __name__ == "__main__":
    main()
    # upload_to_huggingface(
    #     "/home/can/dynamic_representations/artifacts/trained_saes", "canrager/temporalSAEs", temporal_only=True
    # )