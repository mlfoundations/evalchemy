import argparse
import logging
import os
import tempfile

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=30, max=600),
    reraise=True,
)
def upload_shard(dataset, output_dataset, shard_num, num_shards):
    """Push dataset to Hugging Face Hub with automatic retries."""
    api = HfApi()
    # Check if repo exists before creating
    try:
        api.repo_info(repo_id=output_dataset, repo_type="dataset")
        print(f"Repository {output_dataset} already exists")
    except Exception:
        print(f"Creating new repository {output_dataset}")
        api.create_repo(repo_id=output_dataset, repo_type="dataset")

    try:
        # Create temporary file and save the dataset
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            dataset.to_parquet(tmp.name)
            # Format the filename for the shard
            shard_filename = f"train-{shard_num:05d}-of-{num_shards:05d}.parquet"
            # Upload the file
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=shard_filename,
                repo_id=output_dataset,
                repo_type="dataset",
                commit_message=f"Adding shard {shard_num}",
            )

        print(f"Successfully pushed shard {shard_num} to {output_dataset} as {shard_filename}")
    except Exception as e:
        print(f"Push failed for shard {shard_num}, will retry: {str(e)}")
        raise


def process_shard(
    input_dataset: str,
    rank: int,
    global_size: int,
    model_name: str,
    tp: int,
    output_dataset: str = None,
    no_upload: bool = False,
    output_dir: str = None,
    apply_chat_template: bool = True,
) -> None:
    """Process a single shard of the dataset.

    Args:
        input_dataset: The Hugging Face Hub repository ID containing the inputs
        rank: The shard index (0 to global_size-1)
        global_size: Total number of shards
        model_name: The name or path of the model for VLLM
        tp: Tensor parallelism size for VLLM
        output_dataset: Custom output repository ID. If None, a default name will be generated.
        no_upload: If True, don't upload to Hugging Face Hub and save locally instead.
        output_dir: Directory to save the processed data locally when no_upload is True.
                    If no_upload is False, this can also be specified to save a local copy.
        apply_chat_template: Whether to apply the model's chat template to the prompts.
                             Set to False if raw prompts are already formatted.
    """
    # Load the dataset from Hugging Face Hub
    logger.info(f"Loading dataset from {input_dataset}")
    ds = load_dataset(input_dataset, split="train")

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
        tensor_parallel_size=tp,
    )

    # Initialize tokenizer for chat templates if needed
    tokenizer = None
    if apply_chat_template:
        logger.info("Initializing tokenizer for chat template application")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info(
                f"Tokenizer successfully loaded with chat template support: {hasattr(tokenizer, 'apply_chat_template')}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer for chat templates: {e}")
            logger.warning("Proceeding without chat template application")
            apply_chat_template = False
    # llm = None # test

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

        # Apply chat template if needed
        if apply_chat_template and tokenizer and hasattr(tokenizer, "apply_chat_template"):
            prompts = []
            for ex in examples:
                # Extract messages from context
                if "context" in ex and isinstance(ex["context"], list):
                    messages = ex["context"]
                    # Check if messages are already in the right format
                    if all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
                        # Apply chat template
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        prompts.append(formatted_prompt)
                    else:
                        # Fallback to content if messages don't have the right format
                        logger.warning(f"Messages don't have the expected format, falling back to raw content")
                        prompts.append(messages[0]["content"] if len(messages) > 0 else "")
                else:
                    # Fallback if context is not as expected
                    prompts.append("")
            logger.info(f"Applied chat template to {len(prompts)} prompts")
        else:
            # Use raw content as before
            prompts = [ex["context"][0]["content"] for ex in examples]

        logger.info(f"Generating outputs for batch of {len(prompts)} examples with same parameters")
        outputs = llm.generate(prompts, sampling_params[0])  # All sampling_params are the same in the group
        # outputs = [f"This is a test output {i}" for i in range(len(prompts))] # test

        # Process outputs and store results
        for idx, example, output in zip(indices, examples, outputs):
            generated_text = output.outputs[0].text if output.outputs else ""
            # generated_text = "This is a test output for {}".format(idx) # test
            new_example = dict(example)
            new_example["model_outputs"] = generated_text
            processed_examples[idx] = new_example

            # Log progress
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(ds)} examples in shard {rank}")

    # Create a new dataset with the model outputs
    output_ds = Dataset.from_list(processed_examples)
    logger.info(f"Shard successfully processed and loaded into dataset: {len(output_ds)} examples")

    # Extract model name for the output repo ID (use last part of path)
    model_short_name = model_name.split("/")[-1]

    # Use provided output_repo_id or generate default one
    if output_dataset is None:
        output_dataset = f"{input_dataset}_{global_size}shards_{model_short_name}"

    # Format the shard filename
    shard_filename = f"train-{rank:05d}-of-{global_size:05d}.parquet"

    # Save locally if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        local_path = os.path.join(output_dir, shard_filename)
        output_ds.to_parquet(local_path)
        logger.info(f"Saved shard {rank} locally to {local_path}")

    # Upload to HF Hub if not disabled
    if not no_upload:
        try:
            upload_shard(output_ds, output_dataset, rank, global_size)
            logger.info(f"Shard {rank} pushed to hub as {output_dataset}")
            logger.info(f"View the dataset at https://huggingface.co/datasets/{output_dataset}")
        except Exception as e:
            logger.error(f"Failed to push shard {rank} after all retries: {str(e)}")
    else:
        logger.info(f"Skipping upload to Hub (--no_upload flag enabled)")


def main():
    """Parse command line arguments and run the sharded inference."""
    parser = argparse.ArgumentParser(description="Run VLLM inference on a sharded dataset")
    parser.add_argument("--global_size", type=int, required=True, help="Total number of shards")
    parser.add_argument("--rank", type=int, required=True, help="Shard index (0-based)")
    parser.add_argument("--input_dataset", type=str, required=True, help="Hugging Face Hub repository ID")
    parser.add_argument(
        "--output_dataset",
        type=str,
        required=False,
        help="Hugging Face Hub repository ID. If not provided, a default name will be generated.",
    )
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model for VLLM")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size for VLLM")
    parser.add_argument("--no_upload", action="store_true", help="Don't upload results to Hugging Face Hub")
    parser.add_argument("--output_dir", type=str, help="Directory to save the processed data locally")
    parser.add_argument(
        "--no_chat_template", action="store_true", help="Disable applying chat templates to the prompts"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.rank < 0 or args.rank >= args.global_size:
        raise ValueError(f"Rank ({args.rank}) must be between 0 and global_size-1 ({args.global_size-1})")

    if args.no_upload and not args.output_dir:
        raise ValueError("--output_dir is required when --no_upload is specified")

    # Process the shard
    process_shard(
        args.input_dataset,
        args.rank,
        args.global_size,
        args.model_name,
        args.tp,
        args.output_dataset,
        args.no_upload,
        args.output_dir,
        not args.no_chat_template,  # If --no_chat_template is provided, we pass False here
    )


if __name__ == "__main__":
    main()
