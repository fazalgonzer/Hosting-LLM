
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. LOAD MODEL INTO GPU
# ==========================================

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

def gpu_memory():
    return round(torch.cuda.memory_allocated() / 1024**2, 2)

print("\n=== MODEL INFORMATION ===")
print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)
print("Model VRAM:", gpu_memory(), "MB")


# ==========================================
# 2. TOKENIZE PROMPT
# ==========================================

prompt = "Explain artificial intelligence in simple terms."

messages = [
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to("cuda")

input_ids = inputs["input_ids"]

print("\n=== TOKENIZATION ===")
print("Prompt:", prompt)
print("Input tokens:", input_ids.shape[1])


# ==========================================
# 3. RAW INFERENCE
# ==========================================

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    outputs = model(
        **inputs,
        use_cache=True
    )

print("\n=== RAW INFERENCE ===")
print("Logits shape:", outputs.logits.shape)

next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)

print("Next token ID:", next_token_id.item())
print("Next token:", repr(tokenizer.decode(next_token_id)))


# ==========================================
# 4. INSPECT KV CACHE
# ==========================================

cache = outputs.past_key_values

print("\n=== KV CACHE ===")
print("Transformer layers:", len(cache.layers))
print("Key shape:", cache.layers[0].keys.shape)
print("Value shape:", cache.layers[0].values.shape)
print("Cache device:", cache.layers[0].keys.device)

cache_bytes = sum(
    tensor.numel() * tensor.element_size()
    for layer in cache.layers
    for tensor in (layer.keys, layer.values)
)

print("KV cache size:", round(cache_bytes / 1024**2, 3), "MB")


# ==========================================
# 5. ATTENTION CONFIGURATION
# ==========================================

attn = model.model.layers[0].self_attn

print("\n=== ATTENTION CONFIGURATION ===")
print("Layers:", model.config.num_hidden_layers)
print("Query heads:", model.config.num_attention_heads)
print("KV heads:", model.config.num_key_value_heads)
print("Head dimension:",
      model.config.hidden_size // model.config.num_attention_heads)

print("Q weights:", attn.q_proj.weight.shape)
print("K weights:", attn.k_proj.weight.shape)
print("V weights:", attn.v_proj.weight.shape)


# ==========================================
# 6. GENERATE ACTUAL LLM RESPONSE
# ==========================================

print("\n=== LLM GENERATION ===")

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = generated[0][input_ids.shape[1]:]

response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("Prompt:", prompt)
print("Response:", response)
print("Generated tokens:", len(new_tokens))


# ==========================================
# 7. GPU MEMORY USAGE
# ==========================================

print("\n=== GPU MEMORY ===")
print("Allocated:", gpu_memory(), "MB")
print(
    "Peak allocated:",
    round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    "MB"
)
print(
    "Reserved:",
    round(torch.cuda.memory_reserved() / 1024**2, 2),
    "MB"
)

print("\n=== FINISHED ===")
