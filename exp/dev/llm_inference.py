import torch as th
from src.model_utils import load_hf_model
from src.project_config import DEVICE, MODELS_DIR

# model_name = "openai-community/gpt2"
model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "google/gemma-2-2b"
# model_name = "google/gemma-3-12b-pt"
# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "allenai/Llama-3.1-Tulu-3-8B"

model, tokenizer = load_hf_model(
    model_name=model_name,
    cache_dir=MODELS_DIR,
    device=DEVICE,
    quantization_bits=None,
)

# print the norm of the unb


inputs = tokenizer(
    "Hello, world! On a warm sunny day, Teli wanted to", return_tensors="pt"
).to(DEVICE)

with th.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
