import os
import torch
from typing import Optional, List
import matplotlib.pyplot as plt
import einops
from src.project_config import ARTIFACTS_DIR, DEVICE
from src.exp_utils import compute_or_load_svd, load_tokens_of_story, load_activations


def plot_num_components_required_to_reconstruct(
    min_components_required_BPT,
    story_idxs,
    layer_idx,
    reconstruction_thresholds,
    model_name,
    dataset_name="SimpleStories/SimpleStories",
):
    num_stories, num_tokens, _ = min_components_required_BPT.shape

    # Generate filename components
    thresholds_str = "_".join([str(t) for t in reconstruction_thresholds])
    model_str = model_name.split("/")[-1]
    save_fname = f"num_components_required_to_reconstruct_model_{model_str}_reconstruction_thresholds_{thresholds_str}_layer_{layer_idx}"

    fig, ax = plt.subplots(
        figsize=(50, 6)
    )  # Increase figure width for better readability
    for b in range(num_stories):
        for t in range(len(reconstruction_thresholds)):
            ax.plot(
                range(num_tokens),
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

    # If only one story is provided, plot the tokens of the story
    if len(story_idxs) == 1:
        tokens_of_story = load_tokens_of_story(
            dataset_name, story_idxs[0], model_name, omit_BOS_token, num_tokens
        )
        print(f"tokens_of_story: {"".join(tokens_of_story)}")

        ax.set_xticks(range(num_tokens))
        ax.set_xticklabels(
            tokens_of_story, rotation=90, ha="right"
        )  # Rotate labels and align right
        ax.tick_params(
            axis="x", which="major", #pad=20
        )  # Increase spacing between labels

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(ARTIFACTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")
    plt.close()


def plot_mean_components_across_stories(
    min_components_required_BPT,
    layer_idx,
    reconstruction_thresholds,
    model_name,
):
    """
    Plot mean number of PCA components required across all stories with 95% confidence intervals.

    Args:
        min_components_required_BPT: Tensor with shape (batch, position, thresholds)
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

    num_stories, max_pos, _ = min_components_required_BPT.shape

    # Plot for each threshold
    for t_idx, threshold in enumerate(reconstruction_thresholds):
        # All stories have the same length now, so we can calculate mean and CI directly across the batch dimension.
        mean_components = (
            min_components_required_BPT[:, :, t_idx].float().mean(dim=0).cpu()
        )
        std_components = (
            min_components_required_BPT[:, :, t_idx].float().std(dim=0).cpu()
        )

        # 95% confidence interval: 1.96 * std / sqrt(n)
        ci_components = 1.96 * std_components / (num_stories**0.5)

        positions = torch.arange(max_pos)

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
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved mean components across stories plot to {fig_path}")
    plt.close()


if __name__ == "__main__":
    ##### Parameters

    # model_name = "openai-community/gpt2"
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"

    layer_idx = 12

    num_total_stories = 100

    # Choose subset of stories for evaluation
    # These indices correspond to samples from the loaded activations,
    # which differ from the actual story indices in the dataset
    # activation_story_idxs = [0, 3, 4, 7, 14]
    activation_story_idxs = list(range(100))
    # activation_story_idxs = [0]

    reconstruction_thresholds = [0.8]
    if len(reconstruction_thresholds) > 1:
        assert len(activation_story_idxs) == 1

    omit_BOS_token = False
    force_recompute = True

    ##### Load activations and PCA (=centered SVD) results on full dataset of stories

    act_LBPD, dataset_story_idxs = load_activations(
        model_name,
        num_total_stories,
        story_idxs=None,
        omit_BOS_token=omit_BOS_token,
    )
    selected_dataset_story_idxs = [dataset_story_idxs[i] for i in activation_story_idxs]

    U_LbC, S_LC, Vt_LCD, means_LD = compute_or_load_svd(
        act_LBPD, model_name, num_total_stories, force_recompute, layer_idx=layer_idx
    )

    Vt_CD = Vt_LCD[0].to(DEVICE)
    mean_over_all_pos_D = means_LD[0].to(DEVICE)
    num_components, hidden_dim = Vt_CD.shape

    ##### Compute variance explained by top-k PCA components

    # Variance of full representations
    story_BPD = act_LBPD[layer_idx, activation_story_idxs, :, :].to(DEVICE)
    story_centered_BPD = (
        story_BPD - mean_over_all_pos_D
    )  # Center the data wrt. all stories used for PCA
    total_variance_BP = torch.sum(story_centered_BPD**2, dim=-1)

    # Cumulative variance of top-k PCA components, per token
    pca_coeffs_BPC = einops.einsum(
        story_centered_BPD,
        Vt_CD,
        "b p d, c d -> b p c",
    )
    pca_variance_BPC = pca_coeffs_BPC**2
    pca_cumulative_variance_BPC = torch.cumsum(pca_variance_BPC, dim=-1)

    # Compute explained variance
    explained_variance_cumulative_BPC = (
        pca_cumulative_variance_BPC / total_variance_BP[:, :, None]
    ).cpu()

    # Find first k components that meet reconstruction_thresholds for explained variance
    meets_threshold_BPCT = (
        explained_variance_cumulative_BPC[:, :, :, None]
        >= torch.tensor(reconstruction_thresholds)[None, None, None, :]
    )
    min_components_required_BPT = torch.argmax(meets_threshold_BPCT.int(), dim=-2)

    # If no solution is found, set to max number of components
    has_solution_BPT = torch.any(meets_threshold_BPCT, dim=-2)
    min_components_required_BPT.masked_fill_(~has_solution_BPT, num_components)

    ##### Plot results

    plot_num_components_required_to_reconstruct(
        min_components_required_BPT,
        selected_dataset_story_idxs,
        layer_idx,
        reconstruction_thresholds,
        model_name,
    )

    # Plot mean components across stories
    plot_mean_components_across_stories(
        min_components_required_BPT,
        layer_idx,
        reconstruction_thresholds,
        model_name,
    )

    ##### Alternative approaches  to compute cumulative variance

    # NOTE The above approach to use the variance of the PCA coefficients is numerically unstable, since PCA vectors are not exactly orthogonal.
    # However, this has no significant impact on min components required

    # reconstruction_BPD = torch.zeros_like(story_BPD).to(DEVICE)
    # pca_cumulative_variance_BPC = torch.zeros_like(pca_coeffs_BPC).to(DEVICE)

    # for c in range(num_components):
    #     reconstruction_BPD += einops.einsum(
    #         pca_coeffs_BPC[:, :, c], Vt_CD[c], "b p, d -> b p d"
    #     )
    #     pca_cumulative_variance_BPC[:, :, c] = torch.sum(reconstruction_BPD**2, dim=-1)

    # print(f"difference {(pca_cumulative_variance_BPC - pca_cumulative_variance1_BPC).max()}")
    # assert torch.allclose(
    #     pca_cumulative_variance_BPC, pca_cumulative_variance1_BPC, atol=1e-3
    # )

    # NOTE Computing pca_cumulative_variance_BPC in single batch exceeds GPU memory

    # pca_decomposition_BPCD = einops.einsum(
    #     pca_coeffs_BPC,
    #     Vt_CD,
    #     "b p c, c d -> b p c d"
    # )
    # pca_cumulative_reconstruction_BPCD = torch.cumsum(
    #     pca_decomposition_BPCD, dim=-2
    # )
    # pca_cumulative_variance_BPC = torch.sum(pca_cumulative_reconstruction_BPCD**2, dim=-1)
