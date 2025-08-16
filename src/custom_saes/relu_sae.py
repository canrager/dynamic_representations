import json

import torch as th
from huggingface_hub import hf_hub_download

import src.custom_saes.base_sae as base_sae


class ReluSAE(base_sae.BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: th.device,
        dtype: th.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)
        self.d_sae = d_sae

    def encode(self, x: th.Tensor):
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        acts = th.relu(pre_acts)
        return acts

    def decode(self, feature_acts: th.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: th.Tensor):
        f = self.encode(x)
        recon = self.decode(f)

        return base_sae.ForwardOutput(
            sae_out=recon,
            fvu=base_sae.compute_fvu(x, recon),
            latent_acts=f,
            latent_indices=th.ones_like(f) * th.arange(self.cfg.d_sae).to(f.device)
        )

    @th.no_grad()
    def normalize_decoder(self):
        """
        This is useful for doing analysis where e.g. feature activation magnitudes are important.
        If training the SAE using the Anthropic April update, the decoder weights are not normalized.
        The normalization is done in float32 to avoid precision issues.
        """

        original_dtype = self.W_dec.dtype
        self.to(dtype=th.float32)

        # Errors can be relatively large in larger SAEs due to floating point precision
        tolerance = 1e-2

        norms = th.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        print("Decoder vectors are not normalized. Normalizing.")

        test_input = th.randn(10, self.cfg.d_in).to(
            dtype=self.dtype, device=self.device
        )
        initial_output = self(test_input)

        self.W_dec.data /= norms[:, None]

        new_norms = th.norm(self.W_dec, dim=1)

        if not th.allclose(new_norms, th.ones_like(new_norms), atol=tolerance):
            max_norm_diff = th.max(th.abs(new_norms - th.ones_like(new_norms)))
            print(f"Max difference in norms: {max_norm_diff.item()}")
            raise ValueError("Decoder weights are not normalized after normalization")

        self.W_enc *= norms
        self.b_enc *= norms

        new_output = self(test_input)

        max_diff = th.abs(initial_output.sae_out - new_output.sae_out).max()
        print(f"Max difference in output: {max_diff}")

        assert th.allclose(initial_output.sae_out, new_output.sae_out, atol=tolerance)

        self.to(dtype=original_dtype)


def load_dictionary_learning_relu_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: th.device,
    dtype: th.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
) -> ReluSAE:
    assert "ae.pt" in filename

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = th.load(path_to_params, map_location=th.device("cpu"))

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

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = ReluSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)  # type: ignore

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "StandardTrainer":
        sae.cfg.architecture = "standard"
    elif config["trainer"]["trainer_class"] == "PAnnealTrainer":
        sae.cfg.architecture = "p_anneal"
    elif config["trainer"]["trainer_class"] == "StandardTrainerAprilUpdate":
        sae.cfg.architecture = "standard_april_update"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        sae.normalize_decoder()

    return sae


if __name__ == "__main__":
    repo_id = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0104"
    filename = "StandardTrainerAprilUpdate_EleutherAI_pythia-160m-deduped_ctx1024_0104/resid_post_layer_8/trainer_11/ae.pt"
    layer = 8
    device = "cuda" if th.cuda.is_available() else "cpu"
    dtype = th.float32

    model_name = "EleutherAI/pythia-160m-deduped"

    sae = load_dictionary_learning_relu_sae(
        repo_id,
        filename,
        model_name,
        device,  # type: ignore
        dtype,
        layer=layer,
    )
    sae.test_sae(model_name)
