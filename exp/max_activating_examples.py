# Which parts of the decoder vector are used for the novel reconstruction?
from dataclasses import dataclass
import torch as th
import matplotlib.pyplot as plt

from src.model_utils import load_sae
from src.configs import *
from src.plotting_utils import savefig


@dataclass
class ExperimentConfig:
    llm: LLMConfig
    sae: SAEConfig
    env: EnvironmentConfig
    data: DatasetConfig


def experiment(act_BLD, cfg):
    pass

    # What's the effective dim of pred_codes and novel_codes?




def frequency_histogram(act_BLD, cfg):

    B, L, D = act_BLD.shape
    num_tokens = B * L
    num_active_D = (act_BLD > 0).float().sum(dim=(0,1))
    is_inactive_D = num_active_D < 0.5
    num_inactive = is_inactive_D.float().sum()

    act_freq_D = num_active_D[~is_inactive_D] / num_tokens


    bins = th.logspace(th.log10(act_freq_D.min()), th.log10(act_freq_D.max()), 50)
    act_freq_np = act_freq_D.float().cpu().numpy()
    plt.hist(act_freq_np, bins=bins.numpy(), log=True)
    plt.xscale("log")
    plt.xlim(1e-6, 1)
    plt.title(f'num_inactive: {num_inactive}\nmin_freq_possible: {1/num_tokens}')


    savefig(f"act_frequency_histogram_{cfg.sae.name}")


def main():
    cfg = ExperimentConfig(
        # sae=TEMPORAL_SELFTRAIN_SAE_CFG,
        sae=BATCHTOPK_SELFTRAIN_SAE_CFG,
        llm=GEMMA2_LLM_CFG,
        env=ENV_CFG,
        data=WEBTEXT_DS_CFG,
    )

    # # Load SAE
    # sae = load_sae(cfg)
    # print(sae.D.shape)

    # Load activations
    # act_type = os.path.join(cfg.sae.name, "codes")
    act_type = os.path.join(cfg.sae.name, "novel_codes")
    artifacts, _ = load_matching_activations(
        cfg,
        [act_type],
        cfg.env.activations_dir,
        compared_attributes=["llm", "data"],
        verbose=False,
    )
    act_BLD = artifacts[act_type]
    act_BLD = act_BLD.to(cfg.env.device)

    print(f"==>> act_BPD.shape: {act_BLD.shape}")

    # experiment(act_BLD, cfg)
    frequency_histogram(act_BLD, cfg)


if __name__ == "__main__":
    main()
