import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = r"E:\Hosted LLM\models\smollm2-135m"
DATASET_PATH = ROOT / "dataset" / "medical_qa"
OUTPUT_PATH = ROOT / "fine-tuning" / "medical-lora-v3"



num_train_epochs = 3

eval_strategy = "epoch"

# 1. Check GPU
assert torch.cuda.is_available(), "CUDA GPU not detected"
print("GPU:", torch.cuda.get_device_name(0))

# 2. Load existing local model
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    dtype=torch.float32
)

# 3. Load saved dataset
dataset = load_from_disk(str(DATASET_PATH))

train = dataset["train"]
eval_ds = dataset["eval"]

print("Dataset columns:", train.column_names)

# 4. Convert medical Q&A into chat examples
def format_example(example):
    question = example.get("Question", example.get("question"))
    answer = example.get("Answer", example.get("answer"))

    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("Expected Question and Answer text columns")

    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": answer}],
    }

train = train.map(
    format_example,
    remove_columns=train.column_names
)
eval_ds = eval_ds.map(
    format_example,
    remove_columns=eval_ds.column_names
)

print("Training examples:", len(train))
print("Evaluation examples:", len(eval_ds))

# 5. LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 6. Training configuration
training_args = SFTConfig(
    output_dir=str(OUTPUT_PATH),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=1024,
    completion_only_loss=True,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    report_to="none",
    fp16=False,
    bf16=False,
    gradient_checkpointing=False,
    dataloader_num_workers=0,
)

# 7. Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()

# 8. Train
trainer.train()

# 9. Evaluate
metrics = trainer.evaluate()
print("Evaluation loss:", metrics["eval_loss"])

# 10. Save adapter
trainer.model.save_pretrained(str(OUTPUT_PATH))
tokenizer.save_pretrained(str(OUTPUT_PATH))

print("LoRA adapter saved:", OUTPUT_PATH)
