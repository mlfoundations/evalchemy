import argparse
import logging
import os
import sys
from typing import Any, Dict, List

from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model
from tenacity import retry, stop_after_attempt, wait_exponential

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


def process_shard(repo_id: str, rank: int, global_size: int, model_name: str, model_args: str) -> None:
    """Process a single shard of the dataset.

    Args:
        repo_id: The Hugging Face Hub repository ID containing the inputs
        rank: The shard index (0 to global_size-1)
        global_size: Total number of shards
        model_name: The model name for the lm-eval-harness model
        model_args: Additional model arguments as a string
    """
    # Load the dataset from Hugging Face Hub
    logger.info(f"Loading dataset from {repo_id}")
    ds = load_dataset(repo_id, split="train")

    # Shard the dataset
    logger.info(f"Sharding dataset: {global_size} shards, processing shard {rank}")
    ds = ds.shard(num_shards=global_size, index=rank)

    logger.info(f"Using model: {model_name}")

    # Initialize the model from lm-eval-harness
    logger.info("Initializing model through lm-eval-harness")
    model = get_model(model_name)

    # Parse model args - similar to how lm-eval-harness does it
    if model_args:
        model_args_dict = {}
        for arg in model_args.split(","):
            if "=" in arg:
                key, value = arg.split("=", 1)
                # Convert to the right type
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:
                    value = float(value)
                model_args_dict[key] = value
    else:
        model_args_dict = {}

    # Create the model with the parsed args
    lm = model.create_from_arg_string(model_args, model_args_dict)
    logger.info(f"Model initialized: {type(lm).__name__}")

    # Create instances for all examples in the shard
    instances = []
    for idx, example in enumerate(ds):
        # Extract gen_kwargs for this example
        gen_kwargs = example.get("gen_kwargs", {})
        context = example["context"]

        # Create an instance for lm-eval-harness
        instance = Instance(
            args=(context, gen_kwargs),
            task_name=example.get("task_name", "unknown"),
            repeat_idx=example.get("repeat_idx", 0),
            request_idx=example.get("request_idx", idx),
            metadata=example.get("metadata", {}),
        )
        instances.append(instance)

    logger.info(f"Processing {len(instances)} examples in shard {rank} in batch")

    # Generate outputs for all instances at once
    try:
        results = lm.generate_until(instances)
        logger.info(f"Generated {len(results)} outputs for shard {rank}")
    except Exception as e:
        logger.error(f"Error generating outputs: {str(e)}")
        # If generation fails completely, create empty outputs
        results = ["ERROR: Generation failed"] * len(ds)

    # Create processed examples with model outputs
    processed_examples = []
    for idx, (example, model_output) in enumerate(zip(ds, results)):
        new_example = dict(example)
        new_example["model_outputs"] = model_output
        processed_examples.append(new_example)

    # Create a new dataset with the model outputs
    output_ds = load_dataset("dict", data={"train": processed_examples})["train"]

    # Extract model name for the output repo ID (use last part of path if applicable)
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    output_repo_id = f"{repo_id}_{model_short_name}"

    # Push the results to Hub
    try:
        push_to_hub_with_retry(output_ds, output_repo_id, f"shard_{rank}")
        logger.info(f"Shard {rank} pushed to hub as {output_repo_id}")
    except Exception as e:
        logger.error(f"Failed to push shard {rank} after all retries: {str(e)}")


def main():
    """Parse command line arguments and run the sharded inference."""
    parser = argparse.ArgumentParser(description="Run VLLM inference on a sharded dataset using lm-eval-harness")
    parser.add_argument("--global_size", type=int, required=True, help="Total number of shards")
    parser.add_argument("--rank", type=int, required=True, help="Shard index (0-based)")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Hub repository ID")
    parser.add_argument("--model", type=str, default="vllm", help="Model type in lm-eval-harness (default: vllm)")
    parser.add_argument(
        "--model_args",
        type=str,
        default="",
        help="Model arguments as a comma-separated string (e.g. 'pretrained=meta-llama/Llama-2-7b-chat-hf,tensor_parallel_size=1')",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.rank < 0 or args.rank >= args.global_size:
        raise ValueError(f"Rank ({args.rank}) must be between 0 and global_size-1 ({args.global_size-1})")

    # Ensure the model_args includes a 'pretrained' key if not provided
    if args.model == "vllm" and "pretrained=" not in args.model_args:
        logger.error("When using 'vllm' model, you must provide 'pretrained=' in model_args")
        sys.exit(1)

    # Process the shard
    process_shard(args.repo_id, args.rank, args.global_size, args.model, args.model_args)


if __name__ == "__main__":
    main()
