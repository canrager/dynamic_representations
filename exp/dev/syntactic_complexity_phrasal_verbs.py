'''
Plot magnitude of activations for syntactic complexity phrasal verb variations
'''

import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import json
from typing import Optional, List
from collections import defaultdict

from src.project_config import INPUTS_DIR, PLOTS_DIR, MODELS_DIR
from src.exp_utils import compute_or_load_llm_artifacts
from src.model_utils import load_tokenizer

class Config():
    def __init__(self):
        self.debug: bool = True

        # Model
        self.llm_name: str = "meta-llama/Llama-3.1-8B"
        self.layer_idx: int = 22
        self.llm_batch_size: int = 100

        # Dataset
        self.dataset_name: str = "syntactic_complexity_phrasal_verbs_from_template.json"
        self.num_total_stories: int = 100
        self.num_tokens_per_story: Optional[int] = None  # None for all tokens
        self.omit_BOS_token: bool = True
        self.selected_story_idxs: Optional[List[int]] = None
        self.force_recompute: bool = (
            True # True is default. False for quicker iteration.
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


def find_particle_position(tokens_P_decoded, particle):
    """
    Find the position of the particle (second part) of the phrasal verb in the tokenized sentence
    """
    
    for pos, token in enumerate(tokens_P_decoded):
        if token.strip().lower() == particle.lower():
            return pos
        
    for pos, token in enumerate(tokens_P_decoded):
        if token.strip().lower() == f" {particle.lower()}":
            return pos
    
    return None

def find_particle_positions_condensed(condensed_tokens_p, particle):
    particle = particle.lower()

    particle_positions = []
    for pos, token in enumerate(condensed_tokens_p):
        token = token.strip().lower()
        if token == particle or token == f" particle":
            particle_positions.append(pos)

    return enumerate(particle_positions)

def align_variation_to_condensed(act_PD, var_tokens_P, cond_tokens_p):
    """
    take act_P, variation_token_ids_P, master_tokens_strs_p
    """
    p = len(cond_tokens_p)
    act_p = th.zeros(p) * th.nan
    var_cnt = 0

    for pos in range(p):
        if var_cnt < len(var_tokens_P) and cond_tokens_p[pos] == var_tokens_P[var_cnt]:
            act_p[pos] = act_PD[var_cnt]
            var_cnt += 1

    return act_p.cpu().numpy()

def plot_phrasal_verb_grid(mag_variations, tokens_variations, condensed_sentences, dataset, cfg):
    """
    Create grid plot with each subplot showing one sample's phrasal verb variations
    with custom x-axis showing all particle positions
    """
    n_samples = len(mag_variations)
    ncols = math.ceil(math.sqrt(n_samples))
    nrows = math.ceil(n_samples / ncols)

    tokenizer = load_tokenizer(cfg.llm_name, MODELS_DIR)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 5), squeeze=False)
    
    colors = ['blue', 'red']
    labels = ['low dependency length', 'high dependency length']

    for i in range(n_samples):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        
        condensed_P = condensed_sentences[i]
        condensed_P_decoded = [tokenizer.decode(id) for id in tokenizer.encode(condensed_P)]
        particle = dataset[i]["components"]["particle"]

        # Set x-axis ticks and labels
        ax.set_xticks(range(len(condensed_P_decoded)))
        tick_labels = ax.set_xticklabels(condensed_P_decoded, rotation=90, fontsize=6)

        # Plot all variations for this sample and collect particle positions
        num_variations = len(mag_variations[i])
        particle_positions = find_particle_positions_condensed(condensed_P_decoded, particle)
        
        for var_idx in range(num_variations):
            mag_P = mag_variations[i][var_idx]
            tokens_P = tokens_variations[i][var_idx]
            tokens_P_decoded = [tokenizer.decode(token) for token in tokens_P]
            
            # Find actual particle position in this variation
            aligned_mag = align_variation_to_condensed(mag_P, tokens_P_decoded, condensed_P_decoded)
            
            # Plot with NaN values (they won't show up as points)
            ax.plot(range(len(aligned_mag)), aligned_mag, color=colors[var_idx], alpha=0.8, label=labels[var_idx], marker='o')

        # Color all particle positions found
        for var_idx, pos in particle_positions:
            chosen_color = colors[var_idx]  
            print(f'chosen color {chosen_color}')
            tick_labels[pos].set_color(chosen_color)
            tick_labels[pos].set_weight('bold')

        ax.set_title(f"Sample {i}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_samples, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].axis("off")

    fig.tight_layout()
    fig_name = os.path.join(PLOTS_DIR, f"phrasal_verb_grid_custom_{cfg.output_file_str}.png")
    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {fig_name}")
    plt.close()

if __name__ == "__main__":
    cfg = Config()

    # Load JSON dataset with master templates
    with open(INPUTS_DIR / cfg.dataset_name, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    print("First dataset sample:", dataset[0] if dataset else "No data")

    # cat all samples
    all_variations = []
    for sample in dataset:
        all_variations.extend(sample["phrasal_verb_variations"])
        num_variations = len(sample["phrasal_verb_variations"])
    
    # Compute activations for all variations
    acts_LBPD, masks_BP, tokens_BP, dataset_story_idxs = compute_or_load_llm_artifacts(
        cfg, loaded_dataset_sequences=all_variations
    )

    # Calculate magnitudes
    mag_LBPD = th.norm(acts_LBPD, dim=-1)
    mag_BPD = mag_LBPD[cfg.layer_idx]

    # Reshape data back to samples x variations structure
    mag_variations = []
    tokens_variations = []
    condensed_sentences = []
    
    for i, sample in enumerate(dataset):
        sample_mags = []
        sample_tokens = []
        
        for var_idx in range(num_variations):
            global_idx = i * num_variations + var_idx
            sample_mags.append(mag_BPD[global_idx])
            sample_tokens.append(tokens_BP[global_idx])
        
        mag_variations.append(sample_mags)
        tokens_variations.append(sample_tokens)
        condensed_sentences.append(dataset[i]["condensed_sentence"])

    plot_phrasal_verb_grid(mag_variations, tokens_variations, condensed_sentences, dataset, cfg)