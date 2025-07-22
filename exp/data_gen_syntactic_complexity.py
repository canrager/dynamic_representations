from src.project_config import DEVICE, MODELS_DIR
from datasets import load_dataset


# Load the first sentence from the first 100 samples of the dataset

num_total_stories = 100
dataset_name = "SimpleStories/SimpleStories"

split = f"train[:{num_total_stories}]"
all_stories = load_dataset(path=dataset_name, cache_dir=MODELS_DIR, split=split)

first_sentences = []

for story in all_stories:
    text = story["story"]
    fs = text.split(".")[0]
    first_sentences.append(fs)

print(first_sentences)





