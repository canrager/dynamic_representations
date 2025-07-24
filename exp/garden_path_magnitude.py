'''
Plot magnitude of activations for garden path sentences
'''

import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from typing import Optional, List
from collections import defaultdict

from src.project_config import INPUTS_DIR, PLOTS_DIR, MODELS_DIR
from src.exp_utils import compute_or_load_llm_artifacts
from src.model_utils import load_tokenizer

class Config():
    def __init__(self):
        self.debug: bool = True

        # Model
        # self.llm_name = "openai-community/gpt2"
        # self.layer_idx = 6
        self.llm_name: str = "meta-llama/Llama-3.1-8B"
        self.layer_idx: int = 12
        self.llm_batch_size: int = 100

        # Dataset
        self.dataset_name: str = "garden_path_sentences"
        self.num_total_stories: int = 100
        self.num_tokens_per_story: Optional[int] = None  # None for all tokens
        self.omit_BOS_token: bool = True
        self.selected_story_idxs: Optional[List[int]] = None
        self.force_recompute: bool = (
            True  # Leave True per default, False is faster but error prone
        )

        # File names
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        self.input_file_str = (
            f"{dataset_str}_{self.num_total_stories}_{self.num_tokens_per_story}"
        )
        self.output_file_str = (
            self.input_file_str
            + f"_l_{self.layer_idx}"
            + f"_didx_{self.selected_story_idxs}"
            + f"_nobos_{self.omit_BOS_token}"
        )

def plot_magnitude_grid(mag_BPD, tokens_BP, first_diff_idxs, cfg):
    # Determine grid size
    n_stories = mag_BPD.shape[0]
    ncols = math.ceil(math.sqrt(n_stories))
    nrows = math.ceil(n_stories / ncols)

    tokenizer = load_tokenizer(cfg.llm_name, MODELS_DIR)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for i in range(n_stories):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        
        # Plot single line for this story
        mag_PD = mag_BPD[i]
        tokens_P = tokens_BP[i]
        tokens_P = [tokenizer.decode(token) for token in tokens_P]
        ax.plot(mag_PD, color='blue', alpha=0.8)
        
        # Add vertical line at the first differing token index
        ax.axvline(x=first_diff_idxs[i], color='red', linestyle='--', linewidth=1)

        ax.set_xticks(range(len(tokens_P)))
        ax.set_xticklabels(tokens_P, rotation=90, fontsize=6)
        ax.set_title(f"Story {i}")

    # Hide unused subplots
    for j in range(n_stories, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].axis("off")

    fig.tight_layout()
    fig_name = os.path.join(PLOTS_DIR, f"gp_mag_grid_{cfg.output_file_str}.png")
    plt.savefig(fig_name, dpi=200)
    print(f"Saved plot to {fig_name}")
    plt.close()

def plot_aggregated_magnitude(mag_BPD, first_diff_idxs, cfg, window_size=3):
    """
    Create aggregated plot with stories normalized and aligned relative to ambiguity token
    """
    n_stories = mag_BPD.shape[0]
    
    # Find stories that have enough tokens around ambiguity position
    valid_stories = []
    for i in range(n_stories):
        ambiguity_pos = first_diff_idxs[i]
        start_pos = ambiguity_pos - window_size
        end_pos = ambiguity_pos + window_size
        
        if start_pos >= 0 and end_pos < mag_BPD.shape[1]:
            valid_stories.append(i)
    
    if not valid_stories:
        print(f"No stories have enough tokens for window_size={window_size}")
        return
    
    # Normalize within each valid story (z-score normalization)
    normalized_trajectories = []
    for i in valid_stories:
        mag_story = mag_BPD[i]
        normalized_mag = (mag_story - th.mean(mag_story, dim=0, keepdim=True)) / th.std(mag_story, dim=0, keepdim=True)
        
        # Extract window around ambiguity token
        ambiguity_pos = first_diff_idxs[i]
        start_pos = ambiguity_pos - window_size
        end_pos = ambiguity_pos + window_size + 1
        trajectory = normalized_mag[start_pos:end_pos]
        
        normalized_trajectories.append(trajectory)
    
    # Stack trajectories and compute statistics
    trajectories_tensor = th.stack(normalized_trajectories)
    mean_trajectory = th.mean(trajectories_tensor, dim=0)
    std_trajectory = th.std(trajectories_tensor, dim=0)
    
    # Compute 95% confidence interval
    n_stories = len(valid_stories)
    # t-score for 95% CI (approximation for large n, exact for small n would need scipy)
    if n_stories > 30:
        t_score = 1.96  # z-score for large samples
    else:
        # Rough t-scores for common small sample sizes
        t_scores = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26,
                   15: 2.14, 20: 2.09, 25: 2.06, 30: 2.04}
        t_score = t_scores.get(n_stories, 2.04)  # Default to n=30 if not in table
    
    ci_half_width = t_score * std_trajectory / th.sqrt(th.tensor(n_stories, dtype=th.float))
    
    # Convert to numpy for plotting
    positions = range(-window_size, window_size + 1)
    mean_traj_np = mean_trajectory.numpy()
    ci_half_width_np = ci_half_width.numpy()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(positions, mean_traj_np, 'b-', linewidth=2, label='Mean trajectory')
    plt.fill_between(positions, 
                     mean_traj_np - ci_half_width_np, 
                     mean_traj_np + ci_half_width_np, 
                     alpha=0.3, color='blue', label='95% CI')
    
    # Mark the ambiguity position
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Ambiguity token')
    
    plt.xlabel('Position relative to ambiguity token')
    plt.ylabel('Normalized representation magnitude')
    plt.title(f'Aggregated Garden Path Magnitude Analysis (n={len(valid_stories)}/{n_stories} stories)\nLayer {cfg.layer_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    fig_name = os.path.join(PLOTS_DIR, f"gp_mag_aggregated_{cfg.output_file_str}.png")
    plt.savefig(fig_name, dpi=80, bbox_inches='tight')
    print(f"Saved aggregated plot to {fig_name}")
    plt.close()

if __name__ == "__main__":
    cfg = Config()

    # df = pd.read_csv(INPUTS_DIR / "gp_same_len.csv")
    # df = pd.read_csv(INPUTS_DIR / "gp_reading_comp.csv")
    df = pd.read_csv(INPUTS_DIR / "gp_reading_comp_contrastive.csv")
    print(df.head())

    col_names = [
        "Sentence",
        "Sentence_GP",
        "Sentence_Post"
    ]

    sentences = {c: df[c].to_list() for c in col_names}
    chosen_col = "Sentence"
    acts_LBPD, masks_BP, tokens_BP, dataset_story_idxs = compute_or_load_llm_artifacts(cfg, loaded_dataset_sequences=sentences[chosen_col])

    # Process artifacts for the chosen column only
    mag_LBPD = th.norm(acts_LBPD, dim=-1)
    mag_BPD = mag_LBPD[cfg.layer_idx]

    tokenizer = load_tokenizer(cfg.llm_name, MODELS_DIR)
    tokenized_sentences = defaultdict(list)
    num_sentences = len(sentences[chosen_col])

    first_diff_idxs = []
    for i in range(num_sentences):
        tok = {}
        for c in col_names:
            tok[c] = tokenizer.encode(sentences[c][i])

        for first_diff_idx in range(len(tok[chosen_col])):
            if (not tok["Sentence"][first_diff_idx] == tok["Sentence_GP"][first_diff_idx]) or (not tok["Sentence"][first_diff_idx] == tok["Sentence_Post"][first_diff_idx]):
                break
        first_diff_idxs.append(first_diff_idx)

    if cfg.omit_BOS_token:
        first_diff_idxs = [idx-2 for idx in first_diff_idxs]
    else:
        first_diff_idxs = [idx-1 for idx in first_diff_idxs]

    plot_magnitude_grid(mag_BPD, tokens_BP, first_diff_idxs, cfg)
    plot_aggregated_magnitude(mag_BPD, first_diff_idxs, cfg)
