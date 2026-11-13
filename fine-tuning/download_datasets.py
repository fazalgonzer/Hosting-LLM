
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict

MODEL_DATASET = "prithvi1029/medquad-medical-qa"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Stream instead of downloading the full dataset
ds = load_dataset(
    MODEL_DATASET,
    split="train",
    streaming=True
)

# Shuffle and select 600 examples
samples = list(
    ds.shuffle(seed=42, buffer_size=2000).take(600)
)

# Separate training and evaluation data
dataset = DatasetDict({
    "train": Dataset.from_list(samples[:500]),
    "eval": Dataset.from_list(samples[500:])
})

# Save locally for future LoRA training
dataset.save_to_disk(str(DATASET_DIR / "medical_qa"))

print(f"Dataset saved: {DATASET_DIR / 'medical_qa'}")
print(f"Training examples: {len(dataset['train'])}")
print(f"Evaluation examples: {len(dataset['eval'])}")
print(f"Columns: {dataset['train'].column_names}")
print(f"Example: {dataset['train'][0]}")
