import random
from datasets import Dataset
import asyncio 
import secrets
from dcft.external_repositories.airoboros.airoboros.self_instruct import SelfInstructor
import json

def generate_instructions(_: Dataset) -> Dataset:
        random.seed(secrets.randbelow(1000000000))
        setup_dict = {'config_path': 'dcft/external_repositories/airoboros/mini.yaml', 'debug': False}
        self_instructor = SelfInstructor(**setup_dict)
        self_instructor.run()
        temp_dir = self_instructor.temp_dir
        data = []
        breakpoint()
        with open(self_instructor.output_path, "r") as jsonl_file:
            for line in jsonl_file:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")

        # Create a Hugging Face dataset
        dataset = Dataset.from_list(data)
        print("Created Hugging Face dataset:")
        return dataset
