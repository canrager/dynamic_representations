import os
import torch
from src.project_config import ARTIFACTS_DIR
from torch import Tensor
from typing import Tuple

def load_activations(model_name: str, num_stories: int) -> Tuple[Tensor, Tensor]:
    """
    Load activations and attention mask tensors for a given model and number of stories.

    Args:
        model_name: Name of the model (e.g., "openai-community/gpt2")
        num_stories: Number of stories used to generate activations

    Returns:
        tuple: (activations tensor, attention mask tensor)
    """
    model_str = model_name.replace("/", "--")
    acts_save_name = f"activations_{model_str}_simple-stories_first-{num_stories}.pt"
    mask_save_name = f"mask_{model_str}_simple-stories_first-{num_stories}.pt"

    acts_path = os.path.join(ARTIFACTS_DIR, acts_save_name)
    mask_path = os.path.join(ARTIFACTS_DIR, mask_save_name)

    activations = torch.load(acts_path, weights_only=False).to("cpu")
    attention_mask = torch.load(mask_path, weights_only=False).to("cpu")

    # Reshape activations to [L, B*P, D] and mask to [B*P]
    L, B, P, D = activations.shape
    activations = activations.reshape(L, B*P, D)
    attention_mask = attention_mask.reshape(B*P)
    
    # Create a mask for valid positions (where attention_mask == 1)
    valid_positions = attention_mask == 1
    
    # Gather only valid activations
    activations = activations[:, valid_positions, :]
    return activations

if __name__ == "__main__":
    num_stories = 100
    model_name = "openai-community/gpt2"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"

    act_LbD = load_activations(model_name, num_stories)
    print(act_LbD.shape)

    # Next: do PCA
