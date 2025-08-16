import torch as th

import sae_bench.custom_saes.base_sae as base_sae


class IdentitySAE(base_sae.BaseSAE):
    def __init__(
        self,
        d_in: int,
        model_name: str,
        hook_layer: int,
        device: th.device,
        dtype: th.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_in, model_name, hook_layer, device, dtype, hook_name)

        # Override the initialized parameters with identity matrices
        self.W_enc.data = th.eye(d_in).to(dtype=dtype, device=device)
        self.W_dec.data = th.eye(d_in).to(dtype=dtype, device=device)

    def encode(self, x: th.Tensor):
        acts = x @ self.W_enc
        return acts

    def decode(self, feature_acts: th.Tensor):
        return feature_acts @ self.W_dec

    def forward(self, x: th.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


if __name__ == "__main__":
    device = th.device(
        "mps"
        if th.backends.mps.is_available()
        else "cuda"
        if th.cuda.is_available()
        else "cpu"
    )
    dtype = th.float32

    model_name = "pythia-70m-deduped"
    hook_layer = 3
    d_model = 512

    identity = IdentitySAE(d_model, model_name, hook_layer, device, dtype)
    test_input = th.randn(1, 128, d_model, device=device, dtype=dtype)

    encoded = identity.encode(test_input)
    test_output = identity.decode(encoded)

    print(f"L0: {(encoded != 0).sum() / 128}")
    print(f"Diff: {th.abs(test_input - test_output).mean()}")
    assert th.equal(test_input, test_output)
