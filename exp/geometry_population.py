from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
import json
import gc
from tqdm import tqdm
import time
import os
from copy import deepcopy

import torch as th
import einops

from src.configs import *
from src.geometry_utils import get_trace_data, tortuosity


@dataclass
class GeometryPopulationConfig:
    start: int
    end: int
    num_sequences: int
    centering: bool
    normalize: bool
    n_components: int
    act_path: str

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig

    # Position subsampling
    num_p: int | None = None  # Number of positions to subsample (None = use all)
    do_log_scale: bool = False  # Use log scale for position subsampling


def compute_geometry_embeddings(acts_BPD: th.Tensor, cfg: GeometryPopulationConfig):
    """
    Compute UMAP embeddings and tortuosity for activations.

    Args:
        acts_BPD: Activations tensor of shape (B, P, D)
        cfg: Configuration for the geometry experiment

    Returns:
        embeddings: UMAP embeddings of shape (B, L, n_components)
        pos_labels: Position labels of shape (B, L)
        tortuosity_value: Mean tortuosity across sequences
    """
    # Slice to the specified range
    sliced_acts = acts_BPD[: cfg.num_sequences, cfg.start : cfg.end, :]

    # Subsample positions if num_p is specified
    if cfg.num_p is not None:
        seq_len = sliced_acts.shape[1]
        if cfg.do_log_scale:
            log_start = th.log10(th.tensor(0, dtype=th.float) + 1)  # +1 to avoid log(0)
            log_end = th.log10(th.tensor(seq_len - 1, dtype=th.float) + 1)
            log_steps = th.linspace(log_start, log_end, cfg.num_p)
            ps = th.round(10**log_steps - 1).long().clamp(0, seq_len - 1)
        else:
            ps = th.linspace(0, seq_len - 1, cfg.num_p, dtype=th.long)

        # Subsample at selected positions
        sliced_acts = sliced_acts[:, ps, :]

    # Compute UMAP embeddings
    embeddings, pos_labels = get_trace_data(
        sliced_acts, centering=cfg.centering, normalize=cfg.normalize,
        n_components=cfg.n_components
    )

    # Compute tortuosity
    tortuosity_value = tortuosity(embeddings.cpu().numpy()).mean()

    return embeddings, pos_labels, tortuosity_value


def is_user(tokens_BP: th.Tensor) -> th.Tensor:
    """
    Return boolean tensor indicating which tokens are from the user (not the model),
    across multiple turns.

    The user signature is [106, 1645] ('<start_of_turn>', 'user')
    The model signature is [106, 2516] ('<start_of_turn>', 'model')

    Args:
        tokens_BP: Token tensor of shape (B, P)

    Returns:
        is_user_BP: Boolean tensor of shape (B, P) where True = user, False = model
    """
    B, P = tokens_BP.shape
    is_user_BP = th.zeros(B, P, dtype=th.bool)

    for b in range(B):
        tokens = tokens_BP[b]
        current_is_user = False

        p = 0
        while p < P - 1:
            # detect start of turn
            if tokens[p] == 106:  # <start_of_turn>
                if tokens[p + 1] == 1645:  # 'user'
                    current_is_user = True
                elif tokens[p + 1] == 2516:  # 'model'
                    current_is_user = False
            is_user_BP[b, p] = current_is_user
            p += 1

        # set last token (since loop ends at P-2)
        is_user_BP[b, -1] = current_is_user

    return is_user_BP



def single_geometry_experiment(
    cfg: GeometryPopulationConfig, acts_BPD: th.Tensor, tokens_BP: th.tensor
):
    """
    Run a single geometry experiment on given activations.

    Args:
        cfg: Configuration for the experiment
        acts_BPD: Activations tensor of shape (B, P, D)
    """
    # Compute embeddings and metrics
    is_user_BP = is_user(tokens_BP)
    print(f"==>> is_user_BP: {is_user_BP.sum(dim=-1).float().mean()}")
    is_user_BP = is_user_BP[: cfg.num_sequences, cfg.start:cfg.end]
    embeddings, pos_labels, tortuosity_value = compute_geometry_embeddings(acts_BPD, cfg)


    # Prepare results
    results = dict(
        embeddings=embeddings.cpu().numpy().tolist(),
        pos_labels=pos_labels.tolist(),
        is_user_labels=is_user_BP.cpu().numpy().tolist(),
        tortuosity=float(tortuosity_value),
        start=cfg.start,
        end=cfg.end,
    )

    # Add config
    results["config"] = asdict(cfg)

    # Save results
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f"geometry_population_{datetime_str}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)

    print(f"saved results to: {save_path}")
    print(f"  tortuosity: {tortuosity_value:.3f}")

    # Cleanup
    del acts_BPD
    th.cuda.empty_cache()
    gc.collect()


def main():
    configs = get_gemma_act_configs(
        cfg_class=GeometryPopulationConfig,
        start=20,
        end=350,  # Adjust based on your needs
        num_sequences=1000,
        centering=False,
        normalize=False,
        n_components=2,  # Use 2 for 2D UMAP, 3 for 3D UMAP
        # Position subsampling
        num_p=20,  # Number of positions to subsample (None = use all)
        do_log_scale=False,  # Use log scale for position subsampling
        # Artifacts
        env=ENV_CFG,
        # data=ALPACA_DS_CFG,
        data=CHAT_DS_CFG,
        llm=IT_GEMMA2_LLM_CFG,
        sae=None,
        act_paths=(
            (
                [None],
                [
                    "activations",
                ],
            ),
            (
                [GEMMA2_RELU_SAE_CFG, GEMMA2_TOPK_SAE_CFG, GEMMA2_BATCHTOPK_SAE_CFG],
                [
                    "codes",
                ],
            ),
            (
                [GEMMA2_TEMPORAL_SAE_CFG],
                [
                    "novel_codes",
                    "pred_codes",
                ],
            ),
        ),
    )

    for cfg in configs:
        act_path = cfg.act_path
        # if cfg.sae is not None:
        #     act_path = os.path.join(cfg.sae.name, cfg.act_path)

        artifacts, _ = load_matching_activations(
            cfg,
            ["tokens", act_path],
            cfg.env.activations_dir,
            compared_attributes=["llm", "data"],
            verbose=True,
        )

        for key in artifacts:
            if "tokens" == key:
                continue
            acts_BPD = artifacts[key]
            tokens_BP = artifacts["tokens"]
            key_config = deepcopy(cfg)
            key_config.act_path = key
            single_geometry_experiment(key_config, acts_BPD, tokens_BP)
            time.sleep(1)


if __name__ == "__main__":
    main()
