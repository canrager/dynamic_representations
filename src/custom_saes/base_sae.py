from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn
from typing import NamedTuple

import src.custom_saes.custom_sae_config as sae_config


class ForwardOutput(NamedTuple):
    sae_out: torch.Tensor
    fvu: torch.Tensor
    latent_acts: torch.Tensor
    latent_indices: torch.Tensor

class BaseSAE(nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))

        # b_enc and b_dec don't have to be used in the encode/decode methods
        # if your SAE doesn't use biases, leave them as zeros
        # NOTE: core/main.py checks for cosine similarity with b_enc, so it's nice to have the field available
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.cfg = sae_config.CustomSAEConfig(
            model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_name,
            hook_layer=hook_layer,
        )
        self.cfg.dtype = self.dtype.__str__().split(".")[1]
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Decode method must be implemented by child classes")

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Forward method must be implemented by child classes")

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = (
            1e-2 if self.W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5
        )

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(f"Decoder weights are not normalized. Max diff: {max_diff.item()}")
            return False

def compute_fvu(x: torch.Tensor, out: torch.Tensor):
    # Compute fraction of variance unexplained (FVU)
    x_centered = x - x.mean(dim=(0,1))
    x_var = x_centered.pow(2).sum(dim=-1)

    out_centered = out - out.mean(dim=(0,1))
    out_var = out_centered.pow(2).sum(dim=-1)
    fvu = 1 - (x_var / out_var)

    return fvu