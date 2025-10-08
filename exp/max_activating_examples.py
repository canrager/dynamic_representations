# Which parts of the decoder vector are used for the novel reconstruction?
from dataclasses import dataclass
import torch as th
import matplotlib.pyplot as plt

from src.model_utils import load_sae, load_hf_model
from src.exp_utils import load_tokens_of_story
from src.configs import *
from src.plotting_utils import savefig


@dataclass
class ExperimentConfig:
    llm: LLMConfig
    sae: SAEConfig
    env: EnvironmentConfig
    data: DatasetConfig

    sae_act_type: str


def experiment(act_BLS, tokens_BL, tokenizer, cfg):
    # Find top 10 activation values and their B,L,S indices across all latents
    B, L, S = act_BLS.shape

    # Reshape to (B*L, S) to find top activations per latent
    act_reshaped = act_BLS.reshape(B * L, S)

    # Find top 10 activations for each latent
    top_values, top_flat_indices = th.topk(act_reshaped, 10, dim=0)  # Shape: (10, S)

    # Convert flat indices back to B,L coordinates
    top_b_indices = top_flat_indices // L  # Shape: (10, S)
    top_l_indices = top_flat_indices % L  # Shape: (10, S)

    print(f"Top 10 activations for each of {S} latents:")
    print(
        f"Shape of results: B indices {top_b_indices.shape}, L indices {top_l_indices.shape}, values {top_values.shape}"
    )

    top_logits_val_SK, top_logits_idx_SK = get_logit_lens(cfg)

    # Example: show top 10 for first few latents
    for s in range(min(3, S)):
        print(f"\n--------------------Latent {s}:")
        print(f"-------Logit attribution:")
        for k in range(top_logits_val_SK.shape[-1]):
            vocab_str = tokenizer.decode(top_logits_idx_SK[s, k])
            attr_score = top_logits_val_SK[s, k]
            print(f"\t{vocab_str}: {attr_score}")

        
        print(f"-------Max activating examples:")
        for rank in range(10):
            b_idx = top_b_indices[rank, s].item()
            l_idx = top_l_indices[rank, s].item()
            value = top_values[rank, s].item()
            print(f"  Rank {rank+1}: B={b_idx}, L={l_idx}, value={value:.6f}")


            token_strs_L = load_tokens_of_story(
                tokens_BP=tokens_BL,
                story_idx=b_idx,
                model_name=cfg.llm.hf_name,
                omit_BOS_token=False,
                seq_length=None,
                tokenizer=tokenizer,
            )

            # Show only 10 tokens before and after the highlighted token
            start_idx = max(0, l_idx - 10)
            end_idx = min(len(token_strs_L), l_idx + 11)

            highlighted_str = ""
            for i in range(start_idx, end_idx):
                t = token_strs_L[i]
                if i == l_idx:
                    highlighted_str += f"**{t}**"  # Bold markers that survive copy-paste
                else:
                    highlighted_str += t
            print(highlighted_str)
            print("\n")

    return top_b_indices, top_l_indices, top_values


def frequency_histogram(act_BLD, cfg):

    B, L, D = act_BLD.shape
    num_tokens = B * L
    num_active_D = (act_BLD > 0).float().sum(dim=(0, 1))
    is_inactive_D = num_active_D < 0.5
    num_inactive = is_inactive_D.float().sum()

    act_freq_D = num_active_D[~is_inactive_D] / num_tokens

    bins = th.logspace(th.log10(act_freq_D.min()), th.log10(act_freq_D.max()), 50)
    act_freq_np = act_freq_D.float().cpu().numpy()
    plt.hist(act_freq_np, bins=bins.numpy(), log=True)
    plt.xscale("log")
    plt.xlim(1e-6, 1)
    plt.title(f"num_inactive: {num_inactive}\nmin_freq_possible: {1/num_tokens}")

    savefig(f"act_frequency_histogram_{cfg.sae.name}")

def get_logit_lens(cfg):
    # Load SAE
    sae = load_sae(cfg)
    dec_SD = sae.D
    dec_SD = dec_SD / th.norm(dec_SD, dim=-1, keepdim=True)

    # Load model
    llm, tokenizer = load_hf_model(cfg)
    unemb_VD = llm._orig_mod.lm_head.weight

    logit_lens_SV = dec_SD @ unemb_VD.T
    top_logits_val_SK, top_logits_idx_SK = th.topk(logit_lens_SV, k=10, dim=-1)
    return top_logits_val_SK, top_logits_idx_SK


def main():
    cfg = ExperimentConfig(
        sae=TEMPORAL_SELFTRAIN_SAE_CFG,
        # sae=BATCHTOPK_SELFTRAIN_SAE_CFG,
        sae_act_type="novel_codes",
        llm=GEMMA2_LLM_CFG,
        env=ENV_CFG,
        data=WEBTEXT_DS_CFG,
    )

    # Load activations
    act_type = os.path.join(cfg.sae.name, cfg.sae_act_type)
    artifacts, _ = load_matching_activations(
        cfg,
        [act_type, "tokens"],
        cfg.env.activations_dir,
        compared_attributes=["llm", "data"],
        verbose=False,
    )
    act_BLD = artifacts[act_type]
    act_BLD = act_BLD.to(cfg.env.device)

    print(f"==>> act_BPD.shape: {act_BLD.shape}")

    # Load corresponding decoded tokens
    tokens_BL = artifacts["tokens"]
    tokenizer = load_hf_model(cfg, tokenizer_only=True)

    # Run experiment

    experiment(act_BLD, tokens_BL, tokenizer, cfg)
    # frequency_histogram(act_BLD, cfg)


if __name__ == "__main__":
    main()
