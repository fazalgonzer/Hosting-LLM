
from pathlib import Path
from datasets import load_from_disk
import random

ROOT = Path(__file__).resolve().parent.parent
dataset = load_from_disk(str(ROOT / "dataset" / "medical_qa"))

train = dataset["train"]

print("Total training examples:", len(train))

random.seed(42)

for i, example in enumerate(random.sample(list(train), 5), 1):
    question = example["Question"]
    answer = example["Answer"]

    print(f"\n{'=' * 60}")
    print(f"EXAMPLE {i}")
    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer[:1500]}")
    print(f"ANSWER LENGTH: {len(answer.split())} words")
