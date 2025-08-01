import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

import src.custom_saes.base_sae as base_sae


class TopKSAE(base_sae.BaseSAE):
    threshold: torch.Tensor
    k: torch.Tensor

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        use_threshold: bool = False,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert isinstance(k, int) and k > 0
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))

        self.use_threshold = use_threshold
        if use_threshold:
            # Optional global threshold to use during inference. Must be positive.
            self.register_buffer(
                "threshold", torch.tensor(-1.0, dtype=dtype, device=device)
            )

    def encode(self, x: torch.Tensor, return_val_ind: bool = False):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        post_relu_feat_acts_BF = nn.functional.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

        if self.use_threshold:
            if self.threshold < 0:
                raise ValueError(
                    "Threshold is not set. The threshold must be set to use it during inference"
                )
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
            return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)  # type: ignore

        top_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=top_acts_BK
        )
        return encoded_acts_BF, top_acts_BK, top_indices_BK

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        f, latent_acts, latent_indices = self.encode(x, return_val_ind=True)
        out = self.decode(f)

        fvu = base_sae.compute_fvu(x, out)

        return base_sae.ForwardOutput(
            sae_out=out,
            fvu=fvu,
            latent_acts=latent_acts,
            latent_indices=latent_indices
        )


def load_dictionary_learning_topk_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
    use_threshold_at_inference: bool = False,
) -> TopKSAE:
    assert "ae.pt" in filename

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config) as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    assert model_name in config["trainer"]["lm_name"]

    k = config["trainer"]["k"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
        "k": "k",
    }

    if "threshold" in pt_params:
        if use_threshold_at_inference:
            key_mapping["threshold"] = "threshold"
        else:
            del pt_params["threshold"]

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = TopKSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        k=k,
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
        use_threshold=use_threshold_at_inference,
    )

    sae.load_state_dict(renamed_params)  # type: ignore

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "TopKTrainer":
        sae.cfg.architecture = "topk"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder vectors are not normalized. Please normalize them")

    return sae


if __name__ == "__main__":

    repo_id = "canrager/saebench_gemma-2-2b_width-2pow14_date-0107"
    filename = "gemma-2-2b_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_2/ae.pt"
    layer = 12

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model_name = "google/gemma-2-2b"
    hook_name = f"blocks.{layer}.hook_resid_post"

    sae = load_dictionary_learning_topk_sae(
        repo_id,
        filename,
        model_name,
        device,  # type: ignore
        dtype,
        layer=layer,
    )
    sae.test_sae(model_name)
