"""
Script for computing and caching residuals between LLM activations and SAE reconstructions.

For each target folder in artifacts/activations:
1. Load activations.pt
2. For each SAE subfolder, load recons.pt or total_recons.pt
3. Compute residuals = activations - reconstruction
4. Save residuals as .pt file
"""

from dataclasses import dataclass
import os
import torch as th
import gc
from pathlib import Path

from src.configs import *


@dataclass
class ResidualsConfig:
    env: EnvironmentConfig


def compute_residuals_for_target_folder(target_folder: str, cfg: ResidualsConfig):
    """
    Compute residuals for all SAE folders in a target folder.

    Args:
        target_folder: Path to folder containing activations.pt and SAE subfolders
        cfg: Configuration object
    """
    target_path = Path(target_folder)

    # Load activations
    activations_path = target_path / "activations.pt"
    if not activations_path.exists():
        print(f"No activations.pt found in {target_folder}, skipping")
        return

    print(f"Loading activations from {activations_path}")
    activations = th.load(activations_path, map_location='cpu')
    activations = activations.to(cfg.env.device)

    # Find all SAE folders (directories that are not files)
    sae_folders = [item for item in target_path.iterdir()
                   if item.is_dir() and not item.name.startswith('.')]

    for sae_folder in sae_folders:
        print(f"Processing SAE folder: {sae_folder.name}")

        # Check for recons.pt or total_recons.pt
        recons_path = sae_folder / "recons.pt"
        total_recons_path = sae_folder / "total_recons.pt"

        reconstruction = None
        recons_file = None

        if recons_path.exists():
            recons_file = "recons.pt"
            reconstruction = th.load(recons_path, map_location='cpu')
        elif total_recons_path.exists():
            recons_file = "total_recons.pt"
            reconstruction = th.load(total_recons_path, map_location='cpu')
        else:
            print(f"No reconstruction file found in {sae_folder}, skipping")
            continue

        print(f"  Loading reconstruction from {recons_file}")
        reconstruction = reconstruction.to(cfg.env.device)

        # Compute residuals
        residuals = activations - reconstruction

        # Move back to CPU before saving
        residuals = residuals.cpu()

        # Save residuals
        residuals_path = sae_folder / "residuals.pt"
        with open(residuals_path, "wb") as f:
            th.save(residuals, f, pickle_protocol=5)

        print(f"  Saved residuals to {residuals_path}")

        # Clean up GPU memory
        del reconstruction, residuals
        if th.cuda.is_available():
            th.cuda.empty_cache()

    # Clean up activations
    del activations
    if th.cuda.is_available():
        th.cuda.empty_cache()
    gc.collect()


def main():
    cfg = ResidualsConfig(env=ENV_CFG)

    activations_dir = Path(cfg.env.activations_dir)

    if not activations_dir.exists():
        print(f"Activations directory {activations_dir} does not exist")
        return

    # Find all target folders
    target_folders = [item for item in activations_dir.iterdir()
                     if item.is_dir() and not item.name.startswith('.')]

    print(f"Found {len(target_folders)} target folders")

    for target_folder in target_folders:
        print(f"\nProcessing target folder: {target_folder.name}")
        compute_residuals_for_target_folder(str(target_folder), cfg)

    print("\nResiduals computation completed!")


if __name__ == "__main__":
    main()