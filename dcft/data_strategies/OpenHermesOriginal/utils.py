from datasets import Dataset, load_dataset
from typing import Dict


def transform_dataset(example: Dict) -> Dict:
    conversation = example["conversations"]

    # Extract instruction from the first human message
    instruction = next((item["value"] for item in conversation if item["from"] == "human"), None)

    # Extract output from the first gpt message
    output = next((item["value"] for item in conversation if item["from"] == "gpt"), None)

    return {"instruction": instruction, "output": output}


def process_dataset() -> Dataset:
    dataset = load_dataset("teknium/OpenHermes-2.5")["train"]
    transformed_dataset = dataset.map(transform_dataset)

    # Remove the original 'conversations' column
    transformed_dataset = transformed_dataset.remove_columns(["conversations"])

    return transformed_dataset
