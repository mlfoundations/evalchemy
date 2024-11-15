from typing import List, Dict, Optional, Any, Set, Tuple
import random
from tqdm import tqdm
from datasets import Dataset
import openai
from openai import OpenAI
import re
import os
import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

MODEL_NAME: str = "gpt-4o-mini"
MAX_WORKERS: int = 1024


def load_seed_instructions(seed_tasks_path: str) -> Dataset:
    """
    Load seed instruction tasks from text files in the specified directory.

    Args:
        seed_tasks_path (str): Path to the directory containing seed task files.

    Returns:
        Dataset: A HuggingFace Dataset containing the seed instructions.
    """
    datasets_list: List[str] = []
    for i in range(1, 6):
        file_path = os.path.join(seed_tasks_path, f"seed_tasks_{i}.txt")
        with open(file_path, "r") as f:
            content = f.read()
        datasets_list.append(content)
    return Dataset.from_dict({"seeds": datasets_list})


@retry(wait=wait_random(min=1, max=20), stop=stop_after_attempt(5))
def generate_instruction(sample: str) -> str:
    """
    Generate a new instruction using OpenAI's API.

    Args:
        sample (str): The seed instruction to base the generation on.

    Returns:
        str: Generated instruction text.

    Raises:
        Exception: If API call fails after maximum retries.
    """
    prompt = f"{sample}\n\nExample 4\n"
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        top_p=0.99,
    )
    return response.choices[0].message.content.strip()


def generate_instruction_with_retry(sample: str) -> Optional[str]:
    """
    Wrapper function for generate_instruction that handles exceptions.
    """
    try:
        return generate_instruction(sample)
    except Exception as e:
        print(f"Failed to generate instruction after 5 attempts. Skipping. Error: {str(e)}")
        return None


def postprocess_instructions(generated_texts: List[str]) -> List[Dict[str, str]]:
    """
    Process generated instruction texts into structured format.

    Args:
        generated_texts (List[str]): List of raw generated instruction texts.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing structured instruction data.
    """
    processed_samples: List[Dict[str, str]] = []
    for generated_text in generated_texts:
        parts = re.split(r"(Instruction:|Input:|Constraints:)", generated_text)
        if len(parts) < 7:
            continue
        sample = {
            "instruction": parts[2].strip(),
            "input": parts[4].strip(),
            "constraints": parts[6].strip() if len(parts) > 6 else "",
        }
        processed_samples.append(sample)
    return processed_samples


def generate_instructions(seed_dataset: Dataset, num_instructions_to_generate: int = 10) -> Dataset:
    """
    Generate multiple instructions using seed dataset.

    Args:
        seed_dataset (Dataset): Dataset containing seed instructions.
        num_instructions_to_generate (int): Number of instructions to generate.

    Returns:
        Dataset: Dataset containing generated instructions.
    """
    seeds = seed_dataset["seeds"]
    generated_instructions: List[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for _ in range(num_instructions_to_generate):
            sample = random.choice(seeds)
            futures.append(executor.submit(generate_instruction_with_retry, sample))

        with tqdm(total=len(futures), desc="Generating instructions") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    generated_instructions.append(result)
                pbar.update(1)

                if len(generated_instructions) % 10 == 0:
                    print(f"Generated {len(generated_instructions)} instructions")

                if len(generated_instructions) >= num_instructions_to_generate:
                    print(f"Max samples reached: {len(generated_instructions)}")
                    for f in futures:
                        f.cancel()
                    break

    processed_samples = postprocess_instructions(generated_instructions)
    return Dataset.from_list(processed_samples)


def filter_identical_to_seeds(example: Dict[str, str], seed_datasets: List[str]) -> bool:
    """
    Check if an example is identical to any seed dataset example.

    Args:
        example (Dict[str, str]): The example to check.
        seed_datasets (List[str]): List of seed dataset contents.

    Returns:
        bool: True if example is unique, False if identical to a seed.
    """
    for seed_task in seed_datasets:
        examples = seed_task.split("Example")[1:]
        for seed_example in examples:
            parts = re.split(r"(Instruction:|Input:|Constraints:)", seed_example)
            if len(parts) >= 7:
                seed_dict = {
                    "instruction": parts[2].strip(),
                    "input": parts[4].strip(),
                    "constraints": parts[6].strip() if len(parts) > 6 else "",
                }
                if (
                    example["instruction"] == seed_dict["instruction"]
                    and example["input"] == seed_dict["input"]
                    and example["constraints"] == seed_dict["constraints"]
                ):
                    return False
    return True


def filter_duplicates(dataset: Dataset) -> Dataset:
    """
    Remove duplicate examples from dataset based on instruction and input.

    Args:
        dataset (Dataset): Dataset to filter.

    Returns:
        Dataset: Dataset with duplicates removed.
    """
    seen: Set[Tuple[str, str]] = set()
    unique_examples: List[Dict[str, str]] = []
    for example in dataset:
        key = (example["instruction"], example["input"])
        if key not in seen:
            seen.add(key)
            unique_examples.append(example)
    return Dataset.from_list(unique_examples)


def filter_and_dedup(generated_dataset: Dataset, seed_tasks_path: str) -> Dataset:
    """
    Filter out examples identical to seeds and remove duplicates.

    Args:
        generated_dataset (Dataset): Dataset to filter.
        seed_tasks_path (str): Path to seed tasks directory.

    Returns:
        Dataset: Filtered dataset.
    """
    seed_dataset = load_seed_instructions(seed_tasks_path)
    filtered_dataset = generated_dataset.filter(
        lambda example: filter_identical_to_seeds(example, seed_dataset["seeds"])
    )
    filtered_dataset = filter_duplicates(filtered_dataset)
    return filtered_dataset


@retry(wait=wait_random(min=1, max=20), stop=stop_after_attempt(5))
def generate_output(instruction: str, input_text: str, constraints: str) -> str:
    """
    Generate output for given instruction.

    Args:
        instruction (str): The instruction to follow.
        input_text (str): Input text for the instruction.
        constraints (str): Constraints for the generation.

    Returns:
        str: Generated output text.

    Raises:
        Exception: If API call fails after maximum retries.
    """
    prompt = f"""
Instruction: {instruction}
Input: {input_text}
Constraints: {constraints}
Output:
"""
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        top_p=1,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def process_output_with_retry(instruction: str, input_text: str, constraints: str) -> Optional[Dict[str, str]]:
    """
    Process output generation with retry mechanism.

    Args:
        instruction (str): The instruction to follow.
        input_text (str): Input text for the instruction.
        constraints (str): Constraints for the generation.

    Returns:
        Optional[Dict[str, str]]: Dictionary containing the processed output or None if generation fails.
    """
    try:
        output = generate_output(instruction, input_text, constraints)
        return {"instruction": instruction, "input": input_text, "constraints": constraints, "output": output}
    except Exception as e:
        print(f"Failed to generate output after 5 attempts. Skipping. Error: {str(e)}")
        return None


def generate_instruction_output(dataset: Dataset) -> Dataset:
    """
    Generate outputs for all examples in the dataset.

    Args:
        dataset (Dataset): Dataset containing instructions.

    Returns:
        Dataset: Dataset with generated outputs added.
    """
    processed_samples: List[Dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for sample in dataset:
            futures.append(
                executor.submit(
                    process_output_with_retry, sample["instruction"], sample["input"], sample["constraints"]
                )
            )

        with tqdm(total=len(futures), desc="Generating outputs") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    processed_samples.append(result)
                pbar.update(1)

                if len(processed_samples) % 10 == 0:
                    print(f"Generated {len(processed_samples)} outputs")

    return Dataset.from_list(processed_samples)


@retry(wait=wait_random(min=1, max=20), stop=stop_after_attempt(5))
def rephrase_example(instruction: str, input_text: str, constraints: str, output: str) -> Optional[Dict[str, Any]]:
    """
    Rephrase an instruction example and integrate input within an instruction.

    Args:
        instruction (str): Original instruction.
        input_text (str): Input text.
        constraints (str): Constraints for the instruction.
        output (str): Expected output.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing the rephrased example or None if generation fails.
    """
    prompt = f"""
Example 1
Instruction: In this task, you are given an article. Your task is to summarize the article in a sentence.
Input: {{INPUT}}
Alternative formulation: My college roommate asked me what this article means: "{{INPUT}}". So I recapped it in layman's terms:

Example 2
Instruction: This task is about writing a correct answer for the reading comprehension task. Based on the information provided in a given passage…
Input: {{INPUT}}
Alternative formulation: {{INPUT}} Based on the given context, the answer to the question is

Example 3
Instruction: {instruction}.
Input: {{INPUT}}
Alternative formulation:
"""
    try:
        retry = True
        retry_count = 10
        while retry and retry_count > 0:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            rephrased_instruction = response.choices[0].message.content.strip()

            if "{INPUT}" not in rephrased_instruction:
                retry_count -= 1
                print(f"Warning: {{INPUT}} not found in rephrased instruction for: {instruction}. Will retry.")
            else:
                retry = False

        return {
            "original_instruction": instruction,
            "rephrased_instruction": rephrased_instruction,
            "input": input_text,
            "constraints": constraints,
            "output": output,
        }
    except Exception as e:
        print(f"Error in rephrase_example: {str(e)}")
        return None


def rephrase_dataset(dataset: Dataset) -> Dataset:
    """
    Rephrase all examples in the dataset.

    Args:
        dataset (Dataset): Dataset containing instruction examples.

    Returns:
        Dataset: Dataset with rephrased examples.
    """
    rephrased_examples: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for sample in dataset:
            futures.append(
                executor.submit(
                    rephrase_example, sample["instruction"], sample["input"], sample["constraints"], sample["output"]
                )
            )

        with tqdm(total=len(futures), desc="Rephrasing instructions") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rephrased_examples.append(result)
                pbar.update(1)

                if len(rephrased_examples) % 10 == 0:
                    print(f"Rephrased {len(rephrased_examples)} examples")

    structured_samples: List[Dict[str, Any]] = []
    for sample in rephrased_examples:
        if sample is not None:
            structured_sample = {
                "instruction": sample["original_instruction"],
                "instances": [
                    {
                        "instruction_with_input": f"{sample['original_instruction']}\n{sample['input']}",
                        "input": sample["input"],
                        "constraints": sample["constraints"],
                        "output": sample["output"],
                    }
                ],
                "reformulations": [
                    {
                        "instruction": sample["rephrased_instruction"],
                        "instruction_with_input": sample["rephrased_instruction"].replace("{INPUT}", sample["input"]),
                        "input": sample["input"],
                        "output": sample["output"],
                    }
                ],
            }
            structured_samples.append(structured_sample)
        else:
            continue

    return Dataset.from_list(structured_samples)
