# Load garden path sentences
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

from src.project_config import INPUTS_DIR, PLOTS_DIR, MODELS_DIR
from src.exp_utils import compute_or_load_llm_artifacts
from src.model_utils import load_tokenizer


class Config:
    def __init__(self):
        self.debug = True

        # Model
        # self.llm_name = "openai-community/gpt2"
        # self.layer_idx = 6
        self.llm_name = "meta-llama/Llama-3.1-8B"
        self.layer_idx = 12
        self.llm_batch_size = 100

        # Dataset
        self.dataset_name = "garden_path_sentences"
        self.num_total_stories = 100
        self.num_tokens_per_story = None  # None for all tokens
        self.omit_BOS_token = True
        self.selected_story_idxs = None
        self.force_recompute = (
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
# acts_LBPD, masks_BP, tokens_BP, dataset_story_idxs = compute_or_load_llm_artifacts(
#     cfg, loaded_dataset_sequences=sentences
# )
artifacts = {
    c: compute_or_load_llm_artifacts(cfg, loaded_dataset_sequences=sentences[c])
    for c in col_names
}

# Process artifacts for all three sentence types
processed_artifacts = {}
for c in col_names:
    acts_LBPD, masks_BP, tokens_BP, dataset_story_idxs = artifacts[c]
    mag_LBPD = th.norm(acts_LBPD, dim=-1)
    mag_BPD = mag_LBPD[cfg.layer_idx]
    processed_artifacts[c] = {
        'mag_BPD': mag_BPD,
        'tokens_BP': tokens_BP
    }

tokenizer = load_tokenizer(cfg.llm_name, MODELS_DIR)

# Use the first sentence type to determine grid size
first_key = col_names[0]
n_stories = processed_artifacts[first_key]['mag_BPD'].shape[0]
ncols = math.ceil(math.sqrt(n_stories))
nrows = math.ceil(n_stories / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

colors = ['blue', 'red', 'green']
labels = ['Ambiguous', 'GP', 'Post']

for i in range(n_stories):
    row, col = divmod(i, ncols)
    ax = axes[row, col]
    
    # Plot all three lines for this story
    for j, c in enumerate(col_names):
        mag_PD = processed_artifacts[c]['mag_BPD'][i]
        tokens_P = processed_artifacts[c]['tokens_BP'][i]
        tokens_P = [tokenizer.decode(token) for token in tokens_P]
        ax.plot(mag_PD, color=colors[j], label=labels[j], alpha=0.8)
    
    # Use tokens from first sentence type for x-axis labels
    tokens_P = processed_artifacts[first_key]['tokens_BP'][i]
    tokens_P = [tokenizer.decode(token) for token in tokens_P]
    ax.set_xticks(range(len(tokens_P)))
    ax.set_xticklabels(tokens_P, rotation=90, fontsize=6)
    ax.set_title(f"Story {i}")
    if i == 0:  # Add legend only to first subplot
        ax.legend()

# Hide unused subplots
for j in range(n_stories, nrows * ncols):
    row, col = divmod(j, ncols)
    axes[row, col].axis("off")

fig.tight_layout()
fig_name = os.path.join(PLOTS_DIR, f"gp_mag_grid_{cfg.output_file_str}.png")
plt.savefig(fig_name, dpi=80)
print(f"Saved plot to {fig_name}")
plt.close()
