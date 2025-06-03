import os
import torch
from typing import Optional, List
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import einops
from src.project_config import ARTIFACTS_DIR
from src.exp_utils import (
    compute_or_load_svd,
    load_tokens_of_story,
    load_activations,
    load_tokens_of_stories,
)
from tqdm import trange, tqdm
import einops


def plot_interactive_heatmap(
    similarity_matrices_LTPP,  # (num_plot_layers, num_plot_thresholds, N, N)
    flattened_tokens,
    layer_idxs_plot,  # Actual layer numbers/IDs used for plotting, e.g., [0, 8, 16]
    threshold_values_plot,  # e.g., tensor([0.0, 0.1, ...])
    lowest_component_LT_plot,  # (num_plot_layers, num_plot_thresholds) num components for plot indices
    model_str,
    story_idxs,
    save_fname_base,
):
    width = 20  # Base width for plot
    num_plot_layers = similarity_matrices_LTPP.shape[0]
    num_plot_thresholds = similarity_matrices_LTPP.shape[1]

    fig = go.Figure()

    # Create all traces, only first one (0,0) visible initially
    for l_idx in range(num_plot_layers):
        for t_idx in range(num_plot_thresholds):
            visible = l_idx == 0 and t_idx == 0
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrices_LTPP[l_idx, t_idx].cpu().numpy(),
                    x=flattened_tokens,
                    y=flattened_tokens,
                    colorscale="RdBu",
                    zmin=-1,  # Ensure consistent color scaling
                    zmax=1,  # Ensure consistent color scaling
                    colorbar=dict(title="Cosine Similarity"),
                    visible=visible,
                )
            )

    # Helper to create title string
    def get_title_text(l_idx, t_idx):
        layer_val = layer_idxs_plot[l_idx]
        thresh_val = threshold_values_plot[t_idx].item()
        components = lowest_component_LT_plot[l_idx, t_idx].item()
        return (
            f"Token Cosine Similarity - Layer {layer_val}, "
            f"{thresh_val:.1%} Expl. Var. ({components} components)<br>"
            f"Model: {model_str} - Stories: {story_idxs}"
        )

    # Helper to create threshold slider steps (labels depend on selected layer)
    def get_threshold_steps(active_l_idx):
        steps = []
        for t_idx in range(num_plot_thresholds):
            visibility_mask = [False] * (num_plot_layers * num_plot_thresholds)
            visibility_mask[active_l_idx * num_plot_thresholds + t_idx] = True

            thresh_val = threshold_values_plot[t_idx].item()
            components = lowest_component_LT_plot[active_l_idx, t_idx].item()
            label = f"{thresh_val:.0%} ({components} comp.)"

            step = dict(
                method="update",
                args=[
                    {"visible": visibility_mask},
                    {"title.text": get_title_text(active_l_idx, t_idx)},
                ],
                label=label,
            )
            steps.append(step)
        return steps

    # Layer slider steps
    layer_steps = []
    for l_idx in range(num_plot_layers):
        default_t_idx_for_new_layer = (
            0  # Or try to preserve current t_idx if possible, but that's hard.
        )
        # Forcing to 0 is simpler.
        visibility_mask = [False] * (num_plot_layers * num_plot_thresholds)
        visibility_mask[l_idx * num_plot_thresholds + default_t_idx_for_new_layer] = (
            True  # Show (l_idx, 0)
        )

        new_threshold_steps = get_threshold_steps(
            l_idx
        )  # Threshold labels for this l_idx

        layer_step = dict(
            method="update",
            args=[
                {"visible": visibility_mask},  # Updates trace visibility
                {  # Updates layout
                    "title.text": get_title_text(l_idx, default_t_idx_for_new_layer),
                    "sliders[1].steps": new_threshold_steps,  # Update threshold slider labels
                    "sliders[1].active": default_t_idx_for_new_layer,  # Reset threshold slider to 0
                },
            ],
            label=str(layer_idxs_plot[l_idx]),
        )
        layer_steps.append(layer_step)

    initial_threshold_steps = get_threshold_steps(0)  # For layer 0

    fig.update_layout(
        title_text=get_title_text(0, 0),  # Initial title for layer 0, threshold 0
        title_font_size=12,
        xaxis_title_text="Tokens",
        xaxis_title_font_size=10,
        yaxis_title_text="Tokens",
        yaxis_title_font_size=10,
        xaxis_tickangle=-45,
        xaxis_tickfont_size=7,
        xaxis_dtick=1,
        yaxis_tickfont_size=7,
        yaxis_dtick=1,
        width=width * 70,
        height=width * 0.8 * 70 + 100,  # Increased height for two sliders
        margin=dict(l=50, r=50, b=150, t=100, pad=4),  # Increased bottom margin
        xaxis={"domain": [0.25, 1.0]},  # Make space on the left for sliders
        sliders=[
            dict(  # Layer Slider
                active=0,
                currentvalue={"prefix": "Layer: "},
                pad={"t": 10, "b": 10},
                x=0.02,  # Position on the left
                y=0.9,  # Positioned towards the top
                len=0.2,  # Shorter horizontal length
                xanchor="left",
                yanchor="top",  # Anchor to the top
                steps=layer_steps,
            ),
            dict(  # Threshold Slider
                active=0,
                currentvalue={"prefix": "Explained Variance (top PCA components): "},
                pad={"t": 10, "b": 10},
                x=0.02,  # Same horizontal alignment
                y=0.7,  # Positioned below the Layer slider
                len=0.2,  # Same horizontal length
                xanchor="left",
                yanchor="top",  # Anchor to the top
                steps=initial_threshold_steps,
            ),
        ],
    )
    web_dir = "/share/u/can/www"

    html_fig_path = os.path.join(ARTIFACTS_DIR, f"{save_fname_base}.html")
    fig.write_html(html_fig_path)
    print(f"Saved interactive token similarity heatmap to {html_fig_path}")


if __name__ == "__main__":

    # model_name = "openai-community/gpt2"
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"
    model_str = model_name.split("/")[-1]

    num_stories = 100

    force_recompute = False
    do_omit_BOS_token = True
    do_truncate_seq_length = False
    subtract_mean = True
    max_tokens_per_story = None

    # story_idxs = [0, 3]
    story_idxs = [4]

    # Layer indices for plotting
    if "Llama-3.1-8B" in model_name:
        num_layers = 32
        plot_layer_indices_actual = [0, 8, 16, 24, 31]
    elif "gpt2" in model_name:
        num_layers = 12
        plot_layer_indices_actual = [0, 4, 8, 11]
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Thresholds for explained variance
    explained_variance_thresholds = torch.tensor([0.5, 0.75, 0.9])  # e.g. 0.1 to 1.0
    num_thresholds = len(explained_variance_thresholds)

    # Load activations and SVD results on full dataset of stories
    act_LbD, act_LBPD, mask_BP, trunc_seq_length = load_activations(
        model_name=model_name,
        num_stories=num_stories,
        story_idxs=None,  # Load for all stories for SVD
        omit_BOS_token=do_omit_BOS_token,
        truncate_seq_length=do_truncate_seq_length,
        subtract_mean=subtract_mean,
        max_tokens_per_story=max_tokens_per_story,
    )

    # Load SVD results
    U_LbC, S_LC, Vt_LCD, means_LD = compute_or_load_svd(
        act_LbD,
        model_name=model_name,
        num_stories=num_stories,
        force_recompute=force_recompute,
    )

    # Load tokens of stories first, as they define the dimensions of the desired heatmap
    tokens_of_stories = load_tokens_of_stories(
        story_idxs,
        model_name,
        do_omit_BOS_token,
        trunc_seq_length,
        max_tokens_per_story,
    )

    for i, story_idx in enumerate(story_idxs):
        print(f"story_idx: {story_idx}")
        print(f"tokens_of_stories[i]: {tokens_of_stories[i]}")

    flattened_tokens = [token for story in tokens_of_stories for token in story]
    num_total_tokens = len(flattened_tokens)

    # Get the activations for the selected layers
    act_selected_LBPD = act_LBPD[plot_layer_indices_actual, :, :, :]
    act_selected_LBPD = act_selected_LBPD[:, story_idxs, :, :]
    mask_BP = mask_BP[story_idxs, :]

    if max_tokens_per_story is not None:
        act_selected_LBPD = act_selected_LBPD[:, :, :max_tokens_per_story, :]
        mask_BP = mask_BP[:, :max_tokens_per_story]

    batch_mask, pos_mask = torch.where(mask_BP)
    act_selected_LbD = act_selected_LBPD[:, batch_mask, pos_mask, :]

    Vt_LCD = Vt_LCD[plot_layer_indices_actual, :, :]
    num_layers, num_components, hidden_dim = (
        Vt_LCD.shape
    )  # num_components = num_hidden_dim

    projected_LbC = einops.einsum(act_selected_LbD, Vt_LCD, "l b d, l c d -> l b c")
    C_total_components = projected_LbC.shape[2]  # Max components in projected_LbC

    # Compute the explained variance of the projected activations
    total_variance_L = torch.sum(act_selected_LbD**2, dim=(-2, -1))
    projected_variance_LC = torch.sum(projected_LbC**2, dim=-2)
    cumulate_variance_LC = torch.cumsum(projected_variance_LC, dim=-1)
    explained_variance_LC = cumulate_variance_LC / total_variance_L[:, None]

    print(f"cumulate_variance_LC: {cumulate_variance_LC[:, -1]}")
    print(f"total_variance_L: {total_variance_L}")
    assert torch.allclose(
        cumulate_variance_LC[:, -1], total_variance_L, rtol=0.01, atol=0.01
    )

    first_component_exceeding_LCT = (
        explained_variance_LC[:, :, None] > explained_variance_thresholds[None, None, :]
    )
    lowest_component_LT = torch.argmax(first_component_exceeding_LCT.int(), dim=-2)

    # Handle the case where no component exceeds the threshold
    any_component_exceeding_LT = torch.any(first_component_exceeding_LCT, dim=-2)
    lowest_component_LT.masked_fill_(~any_component_exceeding_LT, num_components)

    # Compute similarity matrix with projected activations, for the components that exceed the thresholds
    similarity_matrices_LTPP = torch.zeros(
        (num_layers, num_thresholds, num_total_tokens, num_total_tokens),
        device=projected_LbC.device,  # Added device
        dtype=projected_LbC.dtype,  # Added dtype
    )

    # Corrected loop structure
    for l_idx in tqdm(range(num_layers), desc="Computing similarity matrices by layer"):
        for t_idx in range(num_thresholds):
            # k_value is the component index (0-based) that meets the threshold,
            # or C_total_components if the threshold is not met.
            k_value = lowest_component_LT[l_idx, t_idx].item()

            if (
                k_value < C_total_components
            ):  # Threshold was met by component index k_value
                num_slice_end_idx = k_value + 1  # Slice up to this exclusive end index
            else:  # k_value == C_total_components, threshold was not met
                num_slice_end_idx = C_total_components  # Use all available components

            assert (
                num_slice_end_idx >= 0
            ), f"num_slice_end_idx must be >= 0, but got {num_slice_end_idx}"

            # truncated_bc shape: (num_total_tokens, num_slice_end_idx)
            truncated_bc = projected_LbC[l_idx, :, :num_slice_end_idx]

            norm_truncated_bc = torch.linalg.norm(truncated_bc, dim=-1, keepdim=True)
            # Add small epsilon to prevent division by zero for zero-norm vectors
            truncated_normed_bc = truncated_bc / (norm_truncated_bc + 1e-8)

            # proj_similarity_matrix shape: (num_total_tokens, num_total_tokens)
            proj_similarity_matrix = einops.einsum(
                truncated_normed_bc,
                truncated_normed_bc,
                "b1 c, b2 c -> b1 b2",
            )
            similarity_matrices_LTPP[l_idx, t_idx, :, :] = proj_similarity_matrix

    # Save interactive plot
    story_str = "_".join(map(str, story_idxs))
    # Use a more generic save_fname for the interactive plot, or adjust if specific layer is default
    save_fname = f"interactive_token_similarity_heatmap_{model_str}_stories_{story_str}"
    plot_interactive_heatmap(
        similarity_matrices_LTPP,
        flattened_tokens,
        plot_layer_indices_actual,  # Pass actual layer numbers for labels
        explained_variance_thresholds,
        lowest_component_LT,  # Pass the component counts for labels
        model_str,
        story_idxs,
        save_fname,
    )
