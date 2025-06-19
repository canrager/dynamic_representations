import os
import torch
from typing import Optional, List
import matplotlib.pyplot as plt
import einops
from src.project_config import ARTIFACTS_DIR, DEVICE
from src.exp_utils import compute_or_load_svd, load_tokens_of_story, load_activations


def plot_num_components_required_to_reconstruct(
    min_components_required_BPT,
    mask_BP,
    story_idxs,
    layer_idx,
    reconstruction_thresholds,
    model_name,
    xmax=None,
):
    num_stories = len(story_idxs)

    tokens_of_story = None
    if len(story_idxs) == 1:
        max_tokens_per_story = mask_BP[0].sum()
        tokens_of_story = load_tokens_of_story(
            story_idxs[0], model_name, do_omit_BOS_token, max_tokens_per_story
        )
        print(f"tokens_of_story: {"".join(tokens_of_story)}")

    # Generate filename components
    stories_str = "ALL"  # Using "ALL" as specified in the original code
    thresholds_str = "_".join([str(t) for t in reconstruction_thresholds])
    model_str = model_name.split("/")[-1]
    save_fname = f"num_components_required_to_reconstruct_model_{model_str}_reconstruction_thresholds_{thresholds_str}_stories_{stories_str}_layer_{layer_idx}"

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


def plot_mean_components_across_stories(
    min_components_required_BPT,
    mask_BP,
    story_idxs,
    layer_idx,
    reconstruction_thresholds,
    model_name,
):
    """
    Plot mean number of PCA components required across all stories with 95% confidence intervals.

    Args:
        min_components_required_BPT: Tensor with shape (batch, position, thresholds)
        mask_BP: Mask tensor with shape (batch, position)
        story_idxs: List of story indices
        layer_idx: Layer index for filename
        reconstruction_thresholds: List of threshold values
        model_name: Model name for filename
    """
    # Generate filename
    thresholds_str = "_".join([str(t) for t in reconstruction_thresholds])
    model_str = model_name.split("/")[-1]
    save_fname = f"mean_components_across_stories_model_{model_str}_thresholds_{thresholds_str}_layer_{layer_idx}"

    fig, ax = plt.subplots(figsize=(12, 6))

    # Find maximum position across all stories
    max_pos = mask_BP.sum(dim=-1).max().item()
    num_stories = len(story_idxs)

    # Plot for each threshold
    for t_idx, threshold in enumerate(reconstruction_thresholds):
        mean_components = []
        ci_components = []
        positions = []

        for pos in range(max_pos):
            # Get valid stories for this position
            valid_stories = mask_BP[:, pos].bool()
            num_valid = valid_stories.sum().item()

            if num_valid > 0:  # At least one story has valid token at this position
                valid_components = min_components_required_BPT[
                    valid_stories, pos, t_idx
                ]
                mean_val = valid_components.float().mean().cpu().item()

                if num_valid > 1:
                    std_val = valid_components.float().std().cpu().item()
                    # 95% confidence interval: 1.96 * std / sqrt(n)
                    ci_val = 1.96 * std_val / (num_valid**0.5)
                else:
                    ci_val = 0.0

                mean_components.append(mean_val)
                ci_components.append(ci_val)
                positions.append(pos)

        positions = torch.tensor(positions)
        mean_components = torch.tensor(mean_components)
        ci_components = torch.tensor(ci_components)

        # Plot mean line
        label = f"{threshold*100}% variance threshold"
        ax.plot(positions, mean_components, label=label, linewidth=2)

        # Plot 95% CI band
        ax.fill_between(
            positions,
            mean_components - ci_components,
            mean_components + ci_components,
            alpha=0.2,
        )

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Mean Number of PCA Components Required")
    ax.set_title(
        f"Mean PCA Components Required Across Stories (95% CI) - Layer {layer_idx}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(ARTIFACTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved mean components across stories plot to {fig_path}")
    plt.close()


if __name__ == "__main__":

    # model_name = "openai-community/gpt2"
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"

    num_stories = 100
    # reconstruction_thresholds = [0.8, 0.9]
    reconstruction_thresholds = [0.8]

    # Choose subset of stories for evaluation
    # story_idxs = [0, 3, 4, 7, 14]
    story_idxs = list(range(num_stories))
    layer_idx = 6

    if len(reconstruction_thresholds) > 1:
        assert len(story_idxs) == 1

    force_recompute = True
    # Changing the following parameters will require to force recompute
    do_omit_BOS_token = True
    truncate_to_min_seq_length = False
    max_tokens_per_story = None

    # Load activations and SVD results on full dataset of stories

    act_LbD, act_LBPD, mask_BP = load_activations(
        model_name,
        num_stories,
        story_idxs=None,
        omit_BOS_token=do_omit_BOS_token,
        truncate_to_min_seq_length=truncate_to_min_seq_length,
        truncate_seq_length=max_tokens_per_story,
    )
    U_LbC, S_LC, Vt_LCD, means_LD = compute_or_load_svd(
        act_LbD, model_name, num_stories, force_recompute, layer_idx=layer_idx
    )

    Vt_CD = Vt_LCD[0].to(DEVICE)
    num_components, hidden_dim = Vt_CD.shape
    mean_over_all_pos_D = means_LD[0].to(DEVICE)

    story_BPD = act_LBPD[
        layer_idx, story_idxs, :, :
    ].squeeze().to(DEVICE)  # Stories are of different lengths. We concatenate them to a batch and remove right padding before plotting.


    ## Exp 2: Plot explained variance for a story, x: tokens, y: number of PCA components required to reconstruct thresh% of variance.

    story_centered_BPD = story_BPD - mean_over_all_pos_D
    total_variance_BP = torch.sum(story_centered_BPD**2, dim=-1)

    # Cumulative variance of reconstruction, per token
    pca_coeffs_BPC = einops.einsum(
        story_centered_BPD,
        Vt_CD,
        "b p d, c d -> b p c",
    )

    # Computing in single batch exceeds GPU memory
    # pca_decomposition_BPCD = einops.einsum(
    #     pca_coeffs_BPC,
    #     Vt_CD,
    #     "b p c, c d -> b p c d"
    # )
    # pca_cumulative_reconstruction_BPCD = torch.cumsum(
    #     pca_decomposition_BPCD, dim=-2
    # )
    # pca_cumulative_variance_BPC = torch.sum(pca_cumulative_reconstruction_BPCD**2, dim=-1)
    reconstruction_BPD = torch.zeros_like(story_BPD).to(DEVICE)
    pca_cumulative_variance_BPC = torch.zeros_like(pca_coeffs_BPC).to(DEVICE)
    for c in range(num_components):
        reconstruction_BPD += einops.einsum(
            pca_coeffs_BPC[:, :, c],
            Vt_CD[c],
            'b p, d -> b p d'
        )
        pca_cumulative_variance_BPC[:, :, c] = torch.sum(reconstruction_BPD**2, dim=-1)

    
    # Explained variance
    explained_variance_cumulative_BPC = (
        pca_cumulative_variance_BPC / total_variance_BP[:, :, None]
    ).cpu()

    # Find first component that meets threshold
    meets_threshold_BPCT = (
        explained_variance_cumulative_BPC[:, :, :, None]
        >= torch.tensor(reconstruction_thresholds)[None, None, None, :]
    )
    min_components_required_BPT = torch.argmax(meets_threshold_BPCT.int(), dim=-2)

    # If no solution is found, set to max number of components
    has_solution_BPT = torch.any(meets_threshold_BPCT, dim=-2)
    min_components_required_BPT.masked_fill_(~has_solution_BPT, num_components)



    # Plot results
    plot_num_components_required_to_reconstruct(
        min_components_required_BPT,
        mask_BP[story_idxs],
        story_idxs,
        layer_idx,
        reconstruction_thresholds,
        model_name,
        xmax=None,
    )

    # Plot mean components across stories
    plot_mean_components_across_stories(
        min_components_required_BPT,
        mask_BP[story_idxs],
        story_idxs,
        layer_idx,
        reconstruction_thresholds,
        model_name,
    )


# TODO am I subtracting the mean twice?