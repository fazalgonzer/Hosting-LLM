
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ADAPTER_PATH = r"E:\Hosted LLM\fine-tuning\medical-lora-v3"
MODEL_PATH = r"E:\Hosted LLM\models\smollm2-135m"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=torch.float32
).to("cuda")

model.eval()

question = "What are the common symptoms of diabetes?"

messages = [
    {"role": "user", "content": question}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to("cuda")

# def generate(model):
#     with torch.inference_mode():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=150,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     prompt_length = inputs["input_ids"].shape[1]           # V1

#     return tokenizer.decode(
#         output[0][prompt_length:],
#         skip_special_tokens=True
#     )


def generate(model):
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].shape[1]

    return tokenizer.decode(
        output[0][prompt_length:],
        skip_special_tokens=True
    )

print("\n=== BASE MODEL ===")
print(generate(model))

# Attach trained LoRA weights
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    is_trainable=False,
    local_files_only=True
)

model.eval()

print("\n=== FINE-TUNED MODEL ===")
print(generate(model))
