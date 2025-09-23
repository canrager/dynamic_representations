import torch as th
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from src.configs import *
from exp.pca_exp_var import PCAExpVarConfig
from src.plotting_utils import savefig


def main():
    configs = get_gemma_act_configs(
        cfg_class=PCAExpVarConfig,
        act_paths=(
            (
                [None],
                [
                    "activations",
                    # "surrogate"
                ],
            ),
            (
                [BATCHTOPK_SELFTRAIN_SAE_CFG],
                [
                    # "codes",
                    "recons",
                    # "residuals",
                ],
            ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    # "novel_codes",
                    # "novel_recons",
                    # "pred_codes",
                    # "pred_recons",
                    "total_recons",
                    # "residuals",
                ],
            ),
        ),
        window_sizes=[[
            20,
            34,
            58,
            99,
            169,
            288,
            490,
        ]],  # Choosing tuple so it will not be resulting in separate configs
        selected_context_length=500,
        num_windows=5,
        do_log_spacing=False,
        omit_bos_token=True,
        reconstruction_thresh=0.9,
        min_total_tokens_per_window=19999,
        # Artifacts
        env=ENV_CFG,
        data=WEBTEXT_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=None,  # set by act_paths
        act_path=None,  # set by act_paths
    )

    results = load_results_multiple_configs(
        exp_name="pca_exp_var_",
        source_cfgs=configs,
        target_folder=configs[0].env.results_dir,
        recency_rank=0,
        compared_attributes=None,  # compare all attributes
        verbose=True,
    )
    print(results.keys())


if __name__ == "__main__":
    main()
