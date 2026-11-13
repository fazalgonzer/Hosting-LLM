
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = r"E:\Hosted LLM\models\smollm2-135m"
DATASET_PATH = ROOT / "dataset" / "medical_qa"

# 1. Load tokenizer and original base model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=torch.float32
)

# 2. Load the same evaluation dataset
dataset = load_from_disk(str(DATASET_PATH))
eval_ds = dataset["eval"]

# 3. Use the same format as LoRA training
def format_example(example):
    return {
        "prompt": [
            {"role": "user", "content": example["Question"]}
        ],
        "completion": [
            {"role": "assistant", "content": example["Answer"]}
        ]
    }

eval_ds = eval_ds.map(
    format_example,
    remove_columns=eval_ds.column_names
)

# 4. Same evaluation configuration
config = SFTConfig(
    output_dir=str(ROOT / "fine-tuning" / "base-eval"),
    per_device_eval_batch_size=1,
    max_length=1024,
    completion_only_loss=True,
    fp16=False,
    bf16=False,
    report_to="none",
    dataloader_num_workers=0
)

# 5. Evaluate WITHOUT LoRA
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=eval_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer
)

print("\nEvaluating original SmolLM2-135M...")
metrics = trainer.evaluate()

print("\n===== BASE MODEL RESULTS =====")
print("Evaluation loss:", metrics["eval_loss"])
print("Token accuracy:", metrics.get("eval_mean_token_accuracy"))




# W1114 00:45:58.381000 20428 Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
# Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 272/272 [00:00<00:00, 1703.28it/s]
# Tokenizing train dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 926.16 examples/s]
# Building labels for train dataset: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 3482.54 examples/s]
# Truncating train dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 4832.48 examples/s]
# Dropping fully masked examples from train dataset: 100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 8199.05 examples/s]

# Evaluating original SmolLM2-135M...
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:05<00:00, 16.84it/s]

# ===== BASE MODEL RESULTS =====
# Evaluation loss: 2.473396062850952
# Token accuracy: 0.4782307121157646
# (.venv) PS E:\Hosted LLM> 