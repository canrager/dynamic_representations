import os
import torch
from typing import Optional, List
import matplotlib.pyplot as plt
import einops
from src.project_config import ARTIFACTS_DIR
from src.exp_utils import compute_or_load_svd, load_tokens_of_story, load_activations


def plot_num_components_required_to_reconstruct(
    min_components_required_BPT,
    mask_BP,
    story_idxs,
    layer_idx,
    reconstruction_thresholds,
    save_fname,
    xmax=None,
):
    num_stories = len(story_idxs)

    tokens_of_story = None
    if len(story_idxs) == 1:
        tokens_of_story = load_tokens_of_story(
            story_idxs[0], model_name, do_omit_BOS_token, trunc_seq_length
        )
        print(f"tokens_of_story: {"".join(tokens_of_story)}")

    fig, ax = plt.subplots(
        figsize=(35, 6)
    )  # Increase figure width for better readability
    for b in range(num_stories):
        for t in range(len(reconstruction_thresholds)):
            num_tokens = mask_BP[b].sum()
            ax.plot(
                range(num_tokens),  # assumes right padding
                min_components_required_BPT[b, :num_tokens, t],
                label=f"Story {story_idxs[b]}, {reconstruction_thresholds[t]*100}% explained variance",
            )

    ax.legend()
    ax.set_xlabel("Token Position")
    ylabel = f"Minimum PCA Components required for variance threshold"
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Number of PCA components required to reconstruct variance threshold for stories {story_idxs} in layer {layer_idx}"
    )

    if tokens_of_story is not None:
        ax.set_xticks(range(num_tokens))
        ax.set_xticklabels(
            tokens_of_story, rotation=90, ha="right"
        )  # Rotate labels and align right

    if xmax is not None:
        ax.set_xlim(-2, xmax)

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(ARTIFACTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")
    plt.close()


if __name__ == "__main__":

    # model_name = "openai-community/gpt2"
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"

    num_stories = 100
    reconstruction_thresholds = [0.8, 0.9]
    reconstruction_thresholds = [0.8]

    # Choose subset of stories for evaluation
    story_idxs = [0, 3, 4, 7, 14]
    story_idxs = [4]
    layer_idx = 12

    if len(reconstruction_thresholds) > 1:
        assert len(story_idxs) == 1

    force_recompute = False
    do_omit_BOS_token = True
    do_truncate_seq_length = False

    # Load activations and SVD results on full dataset of stories

    act_LbD, act_LBPD, mask_BP, trunc_seq_length = load_activations(
        model_name,
        num_stories,
        story_idxs=None,
        omit_BOS_token=do_omit_BOS_token,
        truncate_seq_length=do_truncate_seq_length,
    )
    U_LbC, S_LC, Vt_LCD, means_LD = compute_or_load_svd(
        act_LbD, model_name, num_stories, force_recompute
    )

    num_layers, num_components, hidden_dim = Vt_LCD.shape

    

    story_BPD = act_LBPD[
        layer_idx, story_idxs, :, :
    ].squeeze()  # Stories are of different lengths. We concatenate them to a batch and remove right padding before plotting.

    ## Exp 2: Plot explained variance for a story, x: tokens, y: number of PCA components required to reconstruct thresh% of variance.

    # Variance of full representation per token
    story_centered_BPD = (
        story_BPD - means_LD[layer_idx][None, None, :]
    )  # Center the data by subtracting the mean that was used during PCA computation
    total_variance_BP = torch.sum(story_centered_BPD**2, dim=-1)

    # Cumulative variance of projection, per token
    pca_BPC = einops.einsum(
        story_centered_BPD, Vt_LCD[layer_idx, :, :], "b p d, c d -> b p c"
    )
    variance_pca_BPC = pca_BPC**2
    cumulative_variance_BPC = torch.cumsum(variance_pca_BPC, dim=-1)

    # Explained variance
    explained_variance_cumulative_BPC = (
        cumulative_variance_BPC / total_variance_BP[:, :, None]
    )

    # Find first component that meets threshold
    meets_threshold_BPCT = explained_variance_cumulative_BPC[:, :, :, None] >= torch.tensor(reconstruction_thresholds)[None, None, None, :]
    min_components_required_BPT = torch.argmax(meets_threshold_BPCT.int(), dim=-2)

    # If no solution is found, set to max number of components
    has_solution_BPT = torch.any(meets_threshold_BPCT, dim=-2)
    min_components_required_BPT.masked_fill_(~has_solution_BPT, num_components)

    # Plot results
    stories_str = "_".join([str(s) for s in story_idxs])
    thresholds_str = "_".join([str(t) for t in reconstruction_thresholds])
    model_str = model_name.split("/")[-1]
    save_fname = f"num_components_required_to_reconstruct_model_{model_str}_reconstruction_thresholds_{thresholds_str}_stories_{stories_str}_layer_{layer_idx}"
    plot_num_components_required_to_reconstruct(
        min_components_required_BPT,
        mask_BP[story_idxs],
        story_idxs,
        layer_idx,
        reconstruction_thresholds,
        save_fname,
        xmax=None,
    )
