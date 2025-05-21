import torch
from src.model_utils import load_model
from project_config import DEVICE, MODELS_DIR


# model_name = "google/gemma-2-2b"
model_name = "google/gemma-3-12b-pt"
# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "allenai/Llama-3.1-Tulu-3-8B"

model, tokenizer = load_model(
    model_name=model_name,
    cache_dir=MODELS_DIR,
    device=DEVICE,
    quantization_bits=None,
)

# print the norm of the unb


inputs = tokenizer("Hello, world!", return_tensors="pt").to(DEVICE)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))