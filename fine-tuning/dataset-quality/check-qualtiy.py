
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = r"E:\Hosted LLM\dataset\medical_qa"

MODEL_PATH = r"E:\Hosted LLM\models\smollm2-135m"


dataset = load_from_disk(str(DATASET_PATH))
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

for split in ["train", "eval"]:
    total = len(dataset[split])
    suitable = 0
    too_long = 0
    unsuitable = 0

    for example in dataset[split]:
        question = example["Question"].strip()
        answer = example["Answer"].strip()
        words = len(answer.split())

        if not question or not answer or words < 15 or words > 120:
            unsuitable += 1
            continue

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False
        )

        if len(tokens) > 512:
            too_long += 1
        else:
            suitable += 1

    print(f"\n{split.upper()}")
    print("Total:", total)
    print("Suitable:", suitable)
    print("Rejected by word count:", unsuitable)
    print("Rejected by token count:", too_long)
