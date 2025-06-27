import os
import torch
from src.model_utils import load_model, load_tokenizer
from src.project_config import DEVICE, MODELS_DIR, ARTIFACTS_DIR
from datasets import load_dataset
from nnsight import LanguageModel
from tqdm import trange
from typing import List
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding, GPT2Tokenizer


def batch_act_cache(
    model: LanguageModel,
    submodules: List[Module],
    inputs_BP: BatchEncoding,
    hidden_dim: int,
    batch_size: int,
    device: str,
) -> Tensor:
    all_acts_LBPD = torch.zeros(
        (
            len(submodules),
            inputs_BP.shape[0],
            inputs_BP.shape[1],
            hidden_dim,
        )
    )

    for batch_start in trange(
        0, inputs_BP.shape[0], batch_size, desc="Batched Forward"
    ):
        batch_end = batch_start + batch_size
        batch_input_ids = inputs_BP[batch_start:batch_end].to(device)
        batch_mask = torch.ones_like(batch_input_ids)
        batch_inputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_mask,
        }

        with (
            torch.inference_mode(),
            model.trace(batch_inputs, scan=False, validate=False),
        ):
            for l, sm in enumerate(submodules):
                all_acts_LBPD[l, batch_start:batch_end] = sm.output[0].save()

    all_acts_LBPD = all_acts_LBPD.to("cpu")
    return all_acts_LBPD


def collect_stories(dname, num_stories, num_tokens):
    # Use stories with the same amounts of tokens
    # For equal weighting across position and avoiding padding errors
    # NOTE: exact tokenization varies by model, therefore, it can be that different models see different stories

    all_stories = load_dataset(path=dname, cache_dir=MODELS_DIR)["train"]

    inputs_BP = []
    selected_story_idxs = []

    for story_idx, story_item in enumerate(all_stories):
        if len(inputs_BP) >= num_stories:
            break

        story_text = story_item["story"]
        input_ids_P = model.tokenizer(story_text, return_tensors="pt").input_ids

        if input_ids_P.shape[1] >= num_tokens:
            inputs_BP.append(input_ids_P[0, :num_tokens])
            selected_story_idxs.append(story_idx)

    inputs_BP = torch.stack(inputs_BP)

    return inputs_BP, selected_story_idxs


if __name__ == "__main__":
    # Define necessary inputs for batch_act_cache
    num_stories = 100
    num_tokens = 300
    batch_size = 20
    dname = "SimpleStories/SimpleStories"
    device = DEVICE

    # model_name = "openai-community/gpt2" # 10 batches take 1 second on A600
    model_name = "meta-llama/Llama-3.1-8B"  # 10 batches take 1 minute on A600
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"

    # Load model
    model = LanguageModel(
        model_name,
        cache_dir=MODELS_DIR,
        device_map=device,  # Use the defined device
        dispatch=True,
    )

    if "gpt2" in model_name:
        print(model)
        print(model.config)
        hidden_dim = model.config.n_embd
        submodules = [model.transformer.h[l] for l in range(model.config.n_layer)]

        # Language Model loads the AutoTokenizer, which does not use the add_bos_token method.
        model.tokenizer = load_tokenizer(model_name)

    elif "Llama" in model_name:
        print(model)
        print(model.config)
        hidden_dim = model.config.hidden_size
        submodules = [
            model.model.layers[l] for l in range(model.config.num_hidden_layers)
        ]

    # Load dataset
    inputs_BP, selected_story_idxs = collect_stories(dname, num_stories, num_tokens)

    # Call batch_act_cache
    all_acts_LbPD = batch_act_cache(
        model=model,
        submodules=submodules,
        inputs_BP=inputs_BP,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        device=device,
    )

    # Save acts
    model_str = model_name.replace("/", "--")
    acts_save_name = f"activations_{model_str}_simple-stories_first-{num_stories}.pt"
    story_idxs_save_name = (
        f"story_idxs_{model_str}_simple-stories_first-{num_stories}.pt"
    )

    with open(os.path.join(ARTIFACTS_DIR, acts_save_name), "wb") as f:
        torch.save(all_acts_LbPD, f, pickle_protocol=5)
    with open(os.path.join(ARTIFACTS_DIR, story_idxs_save_name), "wb") as f:
        torch.save(selected_story_idxs, f, pickle_protocol=5)
