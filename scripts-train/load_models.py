
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16
).to("cuda")

model.eval()

print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)
print("VRAM allocated:", round(
    torch.cuda.memory_allocated() / 1024**2, 2
), "MB")

prompt = "Explain what an LLM is in one sentence."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        use_cache=True
    )

print("Response:", tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
))
