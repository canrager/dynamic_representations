import os
import torch
from src.model_utils import load_model
from src.project_config import DEVICE, MODELS_DIR, ARTIFACTS_DIR
from datasets import load_dataset
from nnsight import LanguageModel
from tqdm import trange
from typing import List
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding


def batch_act_cache(
    model: LanguageModel,
    submodules: List[Module],
    inputs_bL: BatchEncoding,
    hidden_dim: int,
    batch_size: int,
    device: str,
) -> Tensor:
    all_acts_LbPD = torch.zeros(
        (
            len(submodules),
            inputs_bL.input_ids.shape[0],
            inputs_bL.input_ids.shape[1],
            hidden_dim,
        )
    )

    for batch_start in trange(
        0, inputs_bL.input_ids.shape[0], batch_size, desc="Batched Forward"
    ):
        batch_end = batch_start + batch_size
        batch_inputs = {
            "input_ids": inputs_bL.input_ids[batch_start:batch_end],
            "attention_mask": inputs_bL.attention_mask[batch_start:batch_end],
        }

        with (
            torch.inference_mode(),
            model.trace(batch_inputs, scan=False, validate=False),
        ):
            for l, sm in enumerate(submodules):
                all_acts_LbPD[l, batch_start:batch_end] = sm.output[0].save()

    all_acts_LbPD = all_acts_LbPD.to("cpu")
    return all_acts_LbPD


if __name__ == "__main__":
    # Define necessary inputs for batch_act_cache
    num_stories = 100
    batch_size = 10
    dname = "SimpleStories/SimpleStories"
    device = DEVICE

    # model_name = "openai-community/gpt2"
    # model_name = "google/gemma-3-12b-pt"
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"

    # Load dataset
    all_stories = load_dataset(path=dname, cache_dir=MODELS_DIR)["train"]
    stories = all_stories[:num_stories]["story"]

    # Load model
    model = LanguageModel(
        model_name,
        cache_dir=MODELS_DIR,
        device_map=device,  # Use the defined device
        dispatch=True,
    )

    if "gpt2" in model_name:
        hidden_dim = model.config.n_embd
        submodules = [model.transformer.h[l] for l in range(model.config.n_layer)]
    elif "Llama" in model_name:
        print(model)
        hidden_dim = model.config.hidden_size
        submodules = [model.model.layers[l] for l in range(model.config.num_hidden_layers)]

    # Prepare inputs
    inputs_bL = model.tokenizer(
        stories,
        padding=True,
        padding_side="right",
        truncation=False,
        return_tensors="pt",
    ).to(device)

    # Call batch_act_cache
    all_acts_LbPD = batch_act_cache(
        model=model,
        submodules=submodules,
        inputs_bL=inputs_bL,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        device=device,
    )

    # Save acts and attention masks
    model_str = model_name.replace("/", "--")
    acts_save_name = f"activations_{model_str}_simple-stories_first-{num_stories}.pt"
    mask_save_name = f"mask_{model_str}_simple-stories_first-{num_stories}.pt"

    with open(os.path.join(ARTIFACTS_DIR, acts_save_name), "wb") as f:
        torch.save(all_acts_LbPD, f, pickle_protocol=5)
    with open(os.path.join(ARTIFACTS_DIR, mask_save_name), "wb") as f:
        torch.save(inputs_bL.attention_mask, f, pickle_protocol=5)
