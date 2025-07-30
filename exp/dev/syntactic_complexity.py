import torch as th
import spacy
import json
import os
from typing import List, Dict
from src.exp_utils import compute_or_load_llm_artifacts, compute_centered_svd_single_layer
from src.project_config import INPUTS_DIR
from tqdm import trange

# Load the English language model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("We have to download the model first, via `python -m spacy download en_core_web_sm`")

class Config:
    def __init__(self):
        self.debug = True

        # Model
        # self.llm_name = "openai-community/gpt2"
        # self.layer_idx = 6
        self.llm_name = "meta-llama/Llama-3.1-8B"
        self.layer_idx = 12
        self.llm_batch_size = 100

        # Dataset
        self.dataset_name = "syntactic_complexity_phrasal_verbs_from_template.json"
        self.num_total_stories = 100
        self.num_tokens_per_story = None # None for all tokens
        self.omit_BOS_token = True
        self.selected_story_idxs = None
        self.force_recompute = (
            True  # Leave True per default, False is faster but error prone
        )


        # File names
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        self.input_file_str = f"{dataset_str}_{self.num_total_stories}_{self.num_tokens_per_story}"


def calculate_syntactic_complexity(sentence: str) -> int:
    """
    Calculate syntactic complexity as the sum of all dependency lengths.

    Dependency length is the distance between a dependent token and its head token
    in terms of their positions in the sentence.

    Args:
        sentence: Input sentence to analyze

    Returns:
        Total syntactic complexity (sum of all dependency lengths)
    """
    # Parse the sentence
    doc = nlp(sentence)

    total_complexity = 0

    # For each token, calculate distance to its head
    for token in doc:
        # Skip the root token (which has itself as head)
        if token.head != token:
            # Calculate dependency length as absolute distance
            dependency_length = abs(token.i - token.head.i)
            total_complexity += dependency_length

            # Debug output to show dependencies
            print(
                f"'{token.text}' -> '{token.head.text}' (distance: {dependency_length})"
            )

    return total_complexity


def load_master_templates(dataset_name: str) -> Dict[str, str]:
    """
    Load master templates from the dataset file.
    
    Args:
        dataset_name: Name of the dataset JSON file
        
    Returns:
        Dictionary of master templates
    """
    dataset_path = os.path.join(INPUTS_DIR, dataset_name)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    return dataset.get("master_templates", {})


def apply_template_to_components(template: str, components: Dict[str, str]) -> str:
    """
    Apply a master template to sentence components.
    
    Args:
        template: Master template string with placeholders
        components: Dictionary containing subject, object, time, location, phrasal_verb
        
    Returns:
        Generated sentence using the template
    """
    # Extract verb and particle from phrasal_verb
    phrasal_verb_parts = components["phrasal_verb"].split()
    verb = phrasal_verb_parts[0]
    particle = phrasal_verb_parts[1] if len(phrasal_verb_parts) > 1 else ""
    
    return template.format(
        subject=components["subject"],
        verb=verb,
        particle=particle,
        obj=components["object"],
        location=components["location"],
        time=components["time"]
    )


def compute_intrinsic_dimension_PCA_single_sample(act_PD, cfg):
    P, D = act_PD.shape
    C = D
    T = len(cfg.reconstruction_thresholds)
    min_components_required_mean_PT = -th.ones(P, T)
    min_components_required_ci_PT = -th.ones(P, T)

    for p in trange(P, desc="Computing intrinsic dimension per token"):
        act_pD = act_PD[:p+1, :]
        U_LpC, S_LC, Vt_LCD, mean_train_LD = compute_centered_svd_single_layer(act_pD)




if __name__ == "__main__":
    cfg = Config()

    # Example: Load master templates from dataset
    master_templates = load_master_templates(cfg.dataset_name)
    print("Available master templates:")
    for name, template in master_templates.items():
        print(f"  {name}: {template}")
    
    # Example: Generate sentences using templates
    example_components = {
        "subject": "John",
        "object": "books",
        "time": "yesterday",
        "location": "library",
        "phrasal_verb": "picked up"
    }
    
    print("\nExample sentences generated from templates:")
    for template_name, template in master_templates.items():
        sentence = apply_template_to_components(template, example_components)
        complexity = calculate_syntactic_complexity(sentence)
        print(f"  {template_name}: {sentence} (complexity: {complexity})")

    act_LBPD, masks_BP, tokens_BP, dataset_story_idxs = compute_or_load_llm_artifacts(cfg)
