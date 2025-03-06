import argparse
import logging
import os
from typing import Any, Dict, List

from datasets import load_dataset, Dataset
from tenacity import retry, stop_after_attempt, wait_exponential
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure retry decorator for push_to_hub
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=30, max=600),
    reraise=True,
)
def push_to_hub_with_retry(dataset, repo_id, config_name):
    """Push dataset to Hugging Face Hub with automatic retries."""
    try:
        dataset.push_to_hub(repo_id, config_name=config_name)
        logger.info(f"Successfully pushed {config_name} to {repo_id}")
    except Exception as e:
        logger.error(f"Push failed for {config_name}, will retry: {str(e)}")
        raise


def process_shard(repo_id: str, rank: int, global_size: int, model_name: str) -> None:
    """Process a single shard of the dataset.

    Args:
        repo_id: The Hugging Face Hub repository ID containing the inputs
        rank: The shard index (0 to global_size-1)
        global_size: Total number of shards
        model_name: The name or path of the model for VLLM
    """
    # Load the dataset from Hugging Face Hub
    logger.info(f"Loading dataset from {repo_id}")
    ds = load_dataset(repo_id, split="train")

    # Shard the dataset
    logger.info(f"Sharding dataset: {global_size} shards, processing shard {rank}")
    ds = ds.shard(num_shards=global_size, index=rank)

    logger.info(f"Using model: {model_name}")

    # Initialize VLLM
    logger.info("Initializing VLLM")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )

    # Process each example and group by sampling params
    examples_by_params = {}
    for idx, example in enumerate(ds):
        # Extract gen_kwargs for this example
        gen_kwargs = example.get("gen_kwargs", {})

        # Create SamplingParams from gen_kwargs
        sampling_params = SamplingParams(
            temperature=gen_kwargs.get("temperature", 0.7),
            top_p=gen_kwargs.get("top_p", 0.95),
            max_tokens=gen_kwargs.get("max_new_tokens", 4096),
            stop=gen_kwargs.get("stop", None),
            seed=gen_kwargs.get("seed", None),
        )

        # Create a tuple of sampling parameters to use as dictionary key
        param_key = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.max_tokens,
            tuple(sampling_params.stop) if sampling_params.stop else None,
            sampling_params.seed,
        )

        # Group examples by their sampling parameters
        if param_key not in examples_by_params:
            examples_by_params[param_key] = []
        examples_by_params[param_key].append((idx, example, sampling_params))

    # Process examples in batches with same sampling parameters
    processed_examples = [None] * len(ds)  # Pre-allocate list to maintain order
    for param_key, example_group in examples_by_params.items():
        indices, examples, sampling_params = zip(*example_group)
        prompts = [ex["context"][0]["content"] for ex in examples]
        
        logger.info(f"Generating outputs for batch of {len(prompts)} examples with same parameters")
        outputs = llm.generate(prompts, sampling_params[0])  # All sampling_params are the same in the group

        # Process outputs and store results
        for idx, example, output in zip(indices, examples, outputs):
            generated_text = output.outputs[0].text if output.outputs else ""
            new_example = dict(example)
            new_example["model_outputs"] = generated_text
            processed_examples[idx] = new_example

            # Log progress
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(ds)} examples in shard {rank}")

    # Create a new dataset with the model outputs
    output_ds = Dataset.from_list(processed_examples)

    # Push the results to Hub
    # Extract model name for the output repo ID (use last part of path)
    model_short_name = model_name.split("/")[-1]
    output_repo_id = f"{repo_id}_{global_size}shards_{model_short_name}"
    try:
        push_to_hub_with_retry(output_ds, output_repo_id, f"shard_{rank}")
        logger.info(f"Shard {rank} pushed to hub as {output_repo_id}")
        logger.info(f"View the dataset at https://huggingface.co/datasets/{output_repo_id}")
    except Exception as e:
        logger.error(f"Failed to push shard {rank} after all retries: {str(e)}")


def main():
    """Parse command line arguments and run the sharded inference."""
    parser = argparse.ArgumentParser(description="Run VLLM inference on a sharded dataset")
    parser.add_argument("--global_size", type=int, required=True, help="Total number of shards")
    parser.add_argument("--rank", type=int, required=True, help="Shard index (0-based)")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Hub repository ID")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model for VLLM")

    args = parser.parse_args()

    # Validate arguments
    if args.rank < 0 or args.rank >= args.global_size:
        raise ValueError(f"Rank ({args.rank}) must be between 0 and global_size-1 ({args.global_size-1})")

    # Process the shard
    process_shard(args.repo_id, args.rank, args.global_size, args.model_name)


if __name__ == "__main__":
    main()
