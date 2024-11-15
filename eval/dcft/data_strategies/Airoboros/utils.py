import random
from datasets import Dataset
import asyncio
import secrets
from dcft.external_repositories.airoboros.airoboros.self_instruct import SelfInstructor
import json
import subprocess
import os
import asyncio


def generate_instructions() -> Dataset:
    """
    Runs the airoboros generate-instructions command with a hard-coded config path,
    reads the output, and converts it into a HuggingFace Dataset.

    Returns:
        Dataset: A HuggingFace Dataset containing the generated instructions,
                           or None if the subprocess fails.

    Raises:
        subprocess.CalledProcessError: If the subprocess fails to execute.
        json.JSONDecodeError: If the output file cannot be parsed as JSON.
        IOError: If there's an issue reading the output file.
    """
    # Hard-coded configuration path
    config_path = "dcft/external_repositories/airoboros/example-config.yaml"
    # Run the subprocess
    asyncio.run(SelfInstructor(config_path=config_path).run())
    # result = subprocess.run(["airoboros", "generate-instructions", "--config-path", config_path], text=True, check=True)

    # Read the outputted instructions.jsonl
    with open("dcft/external_repositories/airoboros/instructions.jsonl", "r") as file:
        instructions = [json.loads(line) for line in file]

    os.remove("dcft/external_repositories/airoboros/instructions.jsonl")
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(instructions)
    print(f"Dataset created successfully. Size: {len(dataset)} rows")
    return dataset
