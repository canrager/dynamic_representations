import os
import sys
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


def load_model(
    model_name: str,
    cache_dir: str,
    device: str,
    quantization_bits: int = None,
    tokenizer_only: bool = False,
    verbose: bool = True,
):
    """
    Load a huggingface model and tokenizer.

    Args:
        device (str): Device to load the model on ('auto', 'cuda', 'cpu', etc.)
        cache_dir (str, optional): Directory to cache the downloaded model
    """

    if verbose and cache_dir is not None:
        local_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
        local_path_exists = os.path.exists(local_path)
        if local_path_exists:
            print(f"Model exists in {local_path}")
        else:
            print(f"Model does not exist in {local_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer_only:
        return tokenizer

    # Determine quantization
    torch_dtype = torch.bfloat16

    if quantization_bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        quantization_config_str = "8-bit"
    elif quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        quantization_config_str = "4-bit"
    else:
        quantization_config = None
        quantization_config_str = "no"

    if verbose:
        print(f"Using {quantization_config_str} quantization and bfloat16")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        quantization_config=quantization_config,
        cache_dir=cache_dir,
    )

    # Optimize for inference
    model.eval()
    model = torch.compile(model)

    return model, tokenizer
