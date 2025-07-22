# Load garden path sentences
import pandas as pd
from src.project_config import INPUTS_DIR
from src.exp_utils import compute_or_load_llm_artifacts


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
        self.dataset_name = "garden_path_sentences"
        self.num_total_stories = 100
        self.num_tokens_per_story = None # None for all tokens
        self.omit_BOS_token = True
        self.omit_final_token = True
        self.selected_story_idxs = None
        self.force_recompute = (
            True  # Leave True per default, False is faster but error prone
        )


        # File names
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        self.input_file_str = f"{dataset_str}_{self.num_total_stories}_{self.num_tokens_per_story}"

cfg = Config()

df = pd.read_csv(INPUTS_DIR / "gp_same_len.csv")
print(df.head())

sentences = df["sentence_ambiguous"].tolist()
acts_LBPD, masks_BP, tokens_BP, dataset_story_idxs = compute_or_load_llm_artifacts(cfg, loaded_dataset_sequences=sentences)

print(acts_LBPD.shape)
print(masks_BP.shape)
print(tokens_BP.shape)
print(dataset_story_idxs.shape)

print(acts_LBPD[0, 0, :10])
print(masks_BP[0, :10])







