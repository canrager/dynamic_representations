'''
Plot magnitude of activations for relative clause location variations
'''

import torch as th
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import json
from typing import Optional, List

from src.project_config import INPUTS_DIR, PLOTS_DIR
from src.exp_utils import compute_or_load_llm_artifacts

class Config():
    def __init__(self):
        self.debug: bool = True

        # Model
        self.llm_name: str = "meta-llama/Llama-3.1-8B"
        self.layer_idx: int = 22
        self.llm_batch_size: int = 100

        # Dataset
        self.dataset_name: str = "relative_clause_loc.json"
        self.num_total_stories: int = 100
        self.num_tokens_per_story: Optional[int] = None  # None for all tokens
        self.omit_BOS_token: bool = True
        self.selected_story_idxs: Optional[List[int]] = None
        self.force_recompute: bool = (
            True # True is default. False for quicker iteration.
        )

        # File names
        self.llm_str = self.llm_name.split("/")[-1]
        self.dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        self.input_file_str = (
            f"{self.dataset_str}_{self.llm_str}_{self.num_total_stories}_{self.num_tokens_per_story}"
        )
        self.output_file_str = (
            self.input_file_str
            + f"_l_{self.layer_idx}"
            + f"_didx_{self.selected_story_idxs}"
            + f"_nobos_{self.omit_BOS_token}"
        )

def plot_relative_clause_grid(mag_variations, cfg):
    """
    Create grid plot with each subplot showing one sample's relative clause variations
    with simple position-based x-axis
    """
    n_samples = len(mag_variations)
    ncols = math.ceil(math.sqrt(n_samples))
    nrows = math.ceil(n_samples / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 5), squeeze=False)
    
    colors = ['blue', 'red']
    labels = ['mid_rel_clause', 'late_rel_clause']

    for i in range(n_samples):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        
        # Plot both variations for this sample
        for var_idx in range(2):  # Always 2 variations
            mag_P = mag_variations[i][var_idx]
            
            # Simple position-based x-axis
            positions = list(range(len(mag_P)))
            ax.plot(positions, mag_P.cpu().numpy(), color=colors[var_idx], alpha=0.8, 
                   label=labels[var_idx], marker='o', markersize=3)

        ax.set_title(f"Sample {i}")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Activation Magnitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_samples, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].axis("off")

    fig.tight_layout()
    fig_name = os.path.join(PLOTS_DIR, f"relative_clause_grid_{cfg.output_file_str}.png")
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {fig_name}")
    plt.close()

def plot_aggregated_relative_clause(mag_variations, cfg):
    """
    Create aggregated plot with average and 95% CI across positions
    """
    # Determine max length to pad all sequences
    max_len = max(max(len(mag_variations[i][0]), len(mag_variations[i][1])) for i in range(len(mag_variations)))
    
    # Separate mid_rel_clause and late_rel_clause trajectories
    struct1_trajectories = []
    struct2_trajectories = []
    
    for i in range(len(mag_variations)):
        # Pad sequences to max_len with NaN
        for var_idx, trajectory_list in enumerate([struct1_trajectories, struct2_trajectories]):
            mag_P = mag_variations[i][var_idx].cpu().numpy()
            padded_mag = np.full(max_len, np.nan)
            padded_mag[:len(mag_P)] = mag_P
            trajectory_list.append(padded_mag)
    
    # Convert to numpy arrays
    struct1_array = np.array(struct1_trajectories)
    struct2_array = np.array(struct2_trajectories)
    
    # Compute mean and std, ignoring NaN values
    struct1_mean = np.nanmean(struct1_array, axis=0)
    struct1_std = np.nanstd(struct1_array, axis=0)
    struct2_mean = np.nanmean(struct2_array, axis=0)
    struct2_std = np.nanstd(struct2_array, axis=0)
    
    # Count valid samples at each position
    struct1_n = np.sum(~np.isnan(struct1_array), axis=0)
    struct2_n = np.sum(~np.isnan(struct2_array), axis=0)
    
    # Compute 95% confidence intervals
    def compute_ci(mean_vals, std_vals, n_vals):
        ci_half_width = np.full_like(mean_vals, np.nan)
        for i in range(len(mean_vals)):
            if n_vals[i] > 1:
                if n_vals[i] > 30:
                    t_score = 1.96
                else:
                    t_scores = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26,
                               15: 2.14, 20: 2.09, 25: 2.06, 30: 2.04}
                    t_score = t_scores.get(int(n_vals[i]), 2.04)
                ci_half_width[i] = t_score * std_vals[i] / np.sqrt(n_vals[i])
        return ci_half_width
    
    struct1_ci = compute_ci(struct1_mean, struct1_std, struct1_n)
    struct2_ci = compute_ci(struct2_mean, struct2_std, struct2_n)
    
    # Create the plot
    positions = range(max_len)
    plt.figure(figsize=(12, 6))
    
    # Plot mid_rel_clause
    plt.plot(positions, struct1_mean, 'b-', linewidth=2, label='mid_rel_clause')
    plt.fill_between(positions, 
                     struct1_mean - struct1_ci, 
                     struct1_mean + struct1_ci, 
                     alpha=0.3, color='blue')
    
    # Plot late_rel_clause
    plt.plot(positions, struct2_mean, 'r-', linewidth=2, label='late_rel_clause')
    plt.fill_between(positions, 
                     struct2_mean - struct2_ci, 
                     struct2_mean + struct2_ci, 
                     alpha=0.3, color='red')
    
    plt.xlabel('Token Position')
    plt.ylabel('Activation Magnitude')
    plt.title(f'Aggregated Relative Clause Location Analysis (n={len(mag_variations)} pairs)\\nLayer {cfg.layer_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    fig_name = os.path.join(PLOTS_DIR, f"relative_clause_aggregated_{cfg.output_file_str}.png")
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    print(f"Saved aggregated plot to {fig_name}")
    plt.close()

def plot_label_aggregated_relative_clause(mag_variations, token_labels_BP, cfg):
    """
    Create plot aggregating results by unique labels, separated for mid and late rel clause sentences
    """
    # Collect all unique labels
    all_labels = set()
    for seq_labels in token_labels_BP:
        for label in seq_labels:
            if label and label.strip():  # Skip empty labels
                all_labels.add(label)
    
    unique_labels = sorted(list(all_labels))
    
    # Aggregate magnitudes by label for mid and late versions
    mid_label_mags = {label: [] for label in unique_labels}
    late_label_mags = {label: [] for label in unique_labels}
    
    for i in range(len(mag_variations)):
        # Mid rel clause (index 0)
        mid_seq_idx = i * 2
        mid_mags = mag_variations[i][0].cpu().numpy()
        mid_labels = token_labels_BP[mid_seq_idx]
        
        for token_idx, mag in enumerate(mid_mags):
            if token_idx < len(mid_labels):
                label = mid_labels[token_idx]
                if label and label.strip() and label in unique_labels:
                    mid_label_mags[label].append(mag)
        
        # Late rel clause (index 1)
        late_seq_idx = i * 2 + 1
        late_mags = mag_variations[i][1].cpu().numpy()
        late_labels = token_labels_BP[late_seq_idx]
        
        for token_idx, mag in enumerate(late_mags):
            if token_idx < len(late_labels):
                label = late_labels[token_idx]
                if label and label.strip() and label in unique_labels:
                    late_label_mags[label].append(mag)
    
    # Compute statistics for each label
    mid_stats = {}
    late_stats = {}
    
    for label in unique_labels:
        # Mid stats
        if mid_label_mags[label]:
            mid_values = np.array(mid_label_mags[label])
            mid_stats[label] = {
                'mean': np.mean(mid_values),
                'std': np.std(mid_values),
                'n': len(mid_values)
            }
        
        # Late stats
        if late_label_mags[label]:
            late_values = np.array(late_label_mags[label])
            late_stats[label] = {
                'mean': np.mean(late_values),
                'std': np.std(late_values),
                'n': len(late_values)
            }
    
    # Compute 95% confidence intervals
    def compute_ci(mean_val, std_val, n_val):
        if n_val <= 1:
            return 0
        if n_val > 30:
            t_score = 1.96
        else:
            t_scores = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26,
                       15: 2.14, 20: 2.09, 25: 2.06, 30: 2.04}
            t_score = t_scores.get(int(n_val), 2.04)
        return t_score * std_val / np.sqrt(n_val)
    
    # Filter labels that have data for both conditions
    valid_labels = [label for label in unique_labels 
                   if label in mid_stats and label in late_stats]
    
    if not valid_labels:
        print("No labels found with data for both mid and late conditions")
        return
    
    # Prepare data for plotting
    x_pos = np.arange(len(valid_labels))
    mid_means = [mid_stats[label]['mean'] for label in valid_labels]
    mid_cis = [compute_ci(mid_stats[label]['mean'], mid_stats[label]['std'], mid_stats[label]['n']) 
               for label in valid_labels]
    
    late_means = [late_stats[label]['mean'] for label in valid_labels]
    late_cis = [compute_ci(late_stats[label]['mean'], late_stats[label]['std'], late_stats[label]['n']) 
                for label in valid_labels]
    
    # Create the plot
    offset = 0.1
    plt.figure(figsize=(max(10, len(valid_labels) * 0.8), 6))
    
    plt.errorbar(x_pos - offset, mid_means, yerr=mid_cis, 
                 fmt='o', color='blue', alpha=0.7, capsize=5,
                 label='mid_rel_clause', markersize=8)
    plt.errorbar(x_pos + offset, late_means, yerr=late_cis,
                 fmt='o', color='red', alpha=0.7, capsize=5,
                 label='late_rel_clause', markersize=8)
    
    plt.xlabel('Label')
    plt.ylabel('Mean Activation Magnitude')
    plt.title(f'Activation Magnitude by Label Type\n{cfg.llm_str} Layer {cfg.layer_idx}')
    plt.xticks(x_pos, valid_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add sample sizes as text
    for i, label in enumerate(valid_labels):
        mid_n = mid_stats[label]['n']
        late_n = late_stats[label]['n']
        plt.text(i - offset, mid_means[i] + mid_cis[i] + 0.01, f'n={mid_n}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i + offset, late_means[i] + late_cis[i] + 0.01, f'n={late_n}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    fig_name = os.path.join(PLOTS_DIR, f"relative_clause_by_label_{cfg.output_file_str}.png")
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    print(f"Saved label aggregated plot to {fig_name}")
    plt.close()

if __name__ == "__main__":
    cfg = Config()

    # Load JSON dataset with sentence pairs
    with open(INPUTS_DIR / cfg.dataset_name, 'r') as f:
        dataset = json.load(f)
    
    sentence_pairs = dataset["sentence_pairs"]
    print(f"Loaded {len(sentence_pairs)} sentence pairs")
    print("First sentence pair:", sentence_pairs[0] if sentence_pairs else "No data")

    # Prepare word lists and labels for processing
    all_word_lists = []
    all_word_labels = []
    all_sequences = []
    
    for pair in sentence_pairs:
        # Mid relative clause version
        all_word_lists.append(pair["mid_rel_clause_string"])
        all_word_labels.append(pair["mid_rel_clause_label"])
        all_sequences.append(" ".join(pair["mid_rel_clause_string"]))
        
        # Late relative clause version
        all_word_lists.append(pair["late_rel_clause_string"])
        all_word_labels.append(pair["late_rel_clause_label"])
        all_sequences.append(" ".join(pair["late_rel_clause_string"]))

    print(f"Prepared {len(all_sequences)} sequences with word-level labels")
    
    # Compute activations for all sentences with word-level labels
    acts_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = compute_or_load_llm_artifacts(
        cfg, 
        loaded_dataset_sequences=all_sequences,
        loaded_word_lists=all_word_lists,
        loaded_word_labels=all_word_labels
    )

    # Print some example token-label mappings for verification
    if token_labels_BP and cfg.debug:
        from src.model_utils import load_tokenizer
        from src.project_config import MODELS_DIR
        
        tokenizer = load_tokenizer(cfg.llm_name, MODELS_DIR)
        print("\nExample token-label mappings:")
        
        for seq_idx in range(min(3, len(all_sequences))):
            print(f"\nSequence {seq_idx}: {all_sequences[seq_idx]}")
            print(f"Words: {all_word_lists[seq_idx]}")
            print(f"Labels: {all_word_labels[seq_idx]}")
            print("Token -> Label mapping:")
            
            for token_idx in range(min(12, tokens_BP.shape[1])):
                token_id = tokens_BP[seq_idx, token_idx].item()
                if token_id == tokenizer.pad_token_id:
                    break
                    
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                label = token_labels_BP[seq_idx][token_idx] if token_idx < len(token_labels_BP[seq_idx]) else ""
                print(f"  '{token_text}' -> '{label}'")

    # Calculate magnitudes
    mag_LBPD = th.norm(acts_LBPD, dim=-1)
    mag_BPD = mag_LBPD[cfg.layer_idx]

    # Reshape data back to pairs structure
    mag_variations = []
    
    for i in range(len(sentence_pairs)):
        pair_mags = []
        
        # Each pair has 2 variations
        for var_idx in range(2):
            global_idx = i * 2 + var_idx
            pair_mags.append(mag_BPD[global_idx])
        
        mag_variations.append(pair_mags)

    plot_relative_clause_grid(mag_variations, cfg)
    plot_aggregated_relative_clause(mag_variations, cfg)
    plot_label_aggregated_relative_clause(mag_variations, token_labels_BP, cfg)