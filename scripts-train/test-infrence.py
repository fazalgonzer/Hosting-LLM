import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"E:\Hosted LLM\models\smollm2-135m"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    local_files_only=True
).to("cuda")

model.eval()

print("Model loaded:", next(model.parameters()).device)
print("VRAM:", round(torch.cuda.memory_allocated() / 1024**2), "MB")



prompt = "Explain artificial intelligence"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to("cuda")

with torch.inference_mode():
    outputs = model(
        **inputs,
        use_cache=True
    )

cache = outputs.past_key_values

print("Number of layers:", len(cache.layers))

keys = cache.layers[0].keys
values = cache.layers[0].values

print("Key shape:", keys.shape)
print("Value shape:", values.shape)
print("Cache device:", keys.device)

cache_mb = sum(
    tensor.numel() * tensor.element_size()
    for layer in cache.layers
    for tensor in (layer.keys, layer.values)
) / 1024**2

print("Total KV cache:", round(cache_mb, 3), "MB")