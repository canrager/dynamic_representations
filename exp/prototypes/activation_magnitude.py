import os
import torch
from typing import Optional, List, Literal, Union
import matplotlib.pyplot as plt
import einops
from src.project_config import PLOTS_DIR, DEVICE
from src.exp_utils import (
    compute_or_load_svd,
    load_tokens_of_story,
    compute_or_load_llm_artifacts,
    compute_centered_svd,
)
from tqdm import trange
import numpy as np

if __name__ == "__main__":
    ##### Parameters

    model_name = "openai-community/gpt2"
    # model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"
    layer_idx = 6

    dataset_name = "SimpleStories/SimpleStories"
    # dataset_name = "long_factual_sentences.json"
    # dataset_name = "simple_sentences.json"
    num_tokens_per_story = 100
    num_total_stories = 100
    omit_BOS_token = True

    ##### Load activations

    act_LBPD, dataset_story_idxs, tokens_BP = compute_or_load_llm_artifacts(
        model_name,
        num_total_stories,
        story_idxs=None,
        cfg.omit_BOS_token=omit_BOS_token,
        dataset_name=dataset_name,
    )
    if num_tokens_per_story is not None:
        act_LBPD = act_LBPD[:, :, :num_tokens_per_story, :]
        tokens_BP = tokens_BP[:, :num_tokens_per_story]

    act_BPD = act_LBPD[layer_idx, :, :, :]
    magnitude_BP = act_BPD.norm(dim=-1)
    mean_magnitude_BP = magnitude_BP.mean(dim=0)
    std_magnitude_BP = magnitude_BP.std(dim=0)
    ci = 1.96 * std_magnitude_BP / np.sqrt(magnitude_BP.shape[0])
    mean_magnitude_BP = mean_magnitude_BP.cpu().numpy()
    ci = ci.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_magnitude_BP, label="Mean magnitude (L2)")
    ax.fill_between(
        range(len(mean_magnitude_BP)),
        mean_magnitude_BP - ci,
        mean_magnitude_BP + ci,
        alpha=0.2,
        label="95% CI",
    )
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean magnitude")
    ax.set_title(f"Mean magnitude of activations at layer {layer_idx} of {model_name}")
    ax.legend()

    model_str = model_name.replace("/", "--")
    dataset_str = dataset_name.replace("/", "--")
    fig_name = f"mean_magnitude_layer_{layer_idx}_{model_str}_{dataset_str}.png"
    plt.savefig(os.path.join(PLOTS_DIR, fig_name))
    print(f"Saved figure to {os.path.join(PLOTS_DIR, fig_name)}")
    plt.close()
