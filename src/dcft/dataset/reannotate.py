import argparse
import json
import logging
import os
import uuid
import requests
import yaml

from datetime import datetime
from typing import Tuple
from dcft.dataset.annotators import (
    ANNOTATOR_MAP,
    AnnotatorConfig,
    get_annotator,
    is_gpt_annotator,
)
from dcft.dataset.generation import GenerationConfig
from dcft.dataset.hf import get_dataclass_from_path, HF_DATASET_MAP


def setup_logging(logging_level: int, log_filepath: str = None) -> None:
    """
    Initializes logging for the application.

    This function sets up logging to both a specified file and the console.
    File logging is always set to DEBUG level, while console logging uses
    the specified logging level.

    Args:
        logging_level (int): The logging level for console output (e.g., DEBUG, INFO).
        log_filepath (str, optional): The file path for logging output. If provided, logs will be written to this file.
    """
    if log_filepath:
        # Set up logging to both file and stdout
        file_handler = logging.FileHandler(log_filepath, mode="a")
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)

        logging.basicConfig(
            level=logging.DEBUG,  # Set to DEBUG to capture all levels
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[file_handler, console_handler],
        )
    else:
        logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")


def get_rate_limits(annotator: str) -> Tuple[int, int]:
    """
    Function to get rate limits for a given annotator. Makes a single request to openAI API
    and gets the rate limits from the response headers. These rate limits vary per model
    and are determined by your organization's usage tier. View the following:
    https://platform.openai.com/docs/guides/rate-limits/usage-tiers
    https://platform.openai.com/settings/organization/limits

    Args:
        annotator (str): The annotator for which to get the rate limits.

    Returns:
        Tuple[int, int]: The maximum number of requests and tokens per minute.
    """
    # Send a dummy request to get rate limit information
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
        json={"model": annotator, "messages": []},
    )

    # Extract rate limit information from headers
    max_requests = int(response.headers.get("x-ratelimit-limit-requests", 1500))
    max_tokens = int(response.headers.get("x-ratelimit-limit-tokens", 6250000))

    return max_requests, max_tokens


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the reannotation script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Reannotate the responses to a dataset's instructions")
    parser.add_argument(
        "--annotator",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        choices=list(ANNOTATOR_MAP.keys()),
        help="""Model that generates responses to instructions in the given dataset. By default this is set to the cheapest OpenAI model for development and testing""",
    )
    parser.add_argument("--dataset", type=str, required=True, help="")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="datasets/reannotated",
        choices=list(HF_DATASET_MAP.keys()),
        help="Parent dir to store output json and config under subdir via dataset name",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="datasets/temp",
        help="Parent dir to store jobs file(s) and logs under subdir via dataset name",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from a previous (non-batch) run")
    parser.add_argument(
        "--batch", action="store_true", help="Whether to run in batch mode, available for GPT API annotator only."
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=100,
        help="""The number of requests per batch job.
            The number of batch jobs will be the number of instructions divided by this parameter. 
            No maximum number of batch jobs is documented, however there is a maximum number of tokens that can be in the batch queue (not checked by this code). 
            Batch jobs with up to 50k / 100MB in size are supported, although smaller sizes are suggested to take advantage of more parallelism.""",
    )

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--stop", type=str, default=None, help="parsed with .split(',')")

    # GPT API annotator parameters
    parser.add_argument("--frequency_penalty", type=float, default=0)
    parser.add_argument("--logit_bias", type=str, default=None)
    parser.add_argument("--logprobs", type=bool, default=False)
    parser.add_argument("--top_logprobs", type=int, default=None)
    parser.add_argument("--n", type=int, default=1, help="how many completions to generate for each input")
    parser.add_argument("--presence_penalty", type=float, default=0)
    parser.add_argument("--max_requests_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_minute", type=int, default=None)
    parser.add_argument("--logging_level", type=int, default=logging.INFO, help="Logging level for the application")

    return parser.parse_args()


def main():
    """
    Parsed Args:

    """
    args = parse_arguments()

    # save name is given by dataset and annotator
    save_name = f"{args.dataset.replace('/', '_')}_{args.annotator}"

    # create directories for saving
    temp_sub_dir = os.path.join(args.temp_dir, save_name)
    save_sub_dir = os.path.join(args.save_dir, save_name)
    os.makedirs(temp_sub_dir, exist_ok=True)
    os.makedirs(save_sub_dir, exist_ok=True)

    # setup logging
    setup_logging(args.logging_level, os.path.join(temp_sub_dir, "output.log"))

    # Get rate limits for openAI models
    if is_gpt_annotator(args.annotator):
        if not args.batch:
            rpm, tpm = get_rate_limits(args.annotator)
            if args.max_requests_per_minute is None:
                args.max_requests_per_minute = rpm
                print(f"Automatically set max_requests_per_minute to {rpm}")
            if args.max_tokens_per_minute is None:
                args.max_tokens_per_minute = tpm
                print(f"Automatically set max_tokens_per_minute to {tpm}")

    # Load data
    data = get_dataclass_from_path(args.dataset)
    assert len(data.system_prompts) == len(data.user_prompts)
    logging.info(f"Reannotating {len(data.system_prompts)} samples")

    # Setup config objections needed for the annotation pipeline
    annotator_config = AnnotatorConfig(
        annotator_name=args.annotator,
        resume=args.resume,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        batch=args.batch,
        max_batch_size=args.max_batch_size,
    )
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_tokens,
        stop=args.stop,
        frequency_penalty=args.frequency_penalty,
        logit_bias=args.logit_bias,
        logprobs=args.logprobs,
        top_logprobs=args.top_logprobs,
        n=args.n,
        presence_penalty=args.presence_penalty,
    )

    # Run the annotation pipeline
    annotator = get_annotator(args.annotator, annotator_config)
    annotator.annotate(data, generation_config, temp_sub_dir)

    # Save outputs
    save_sub_dir = os.path.join(args.save_dir, save_name)
    os.makedirs(save_sub_dir, exist_ok=True)

    # Save batch objects metadata or the reannotated examples themselves
    if args.batch:
        assert data.batch_objects is not None
        batch_objects_file = os.path.join(temp_sub_dir, "batch_objects.json")
        with open(batch_objects_file, "w") as f:
            json.dump([obj.model_dump() for obj in data.batch_objects], f, indent=4)
        logging.info(f"Batch objects saved to {batch_objects_file}")
        logging.info(
            f"Run `python -m dcft.dataset.watch_gpt_batch --batch_objects_file {batch_objects_file} "
            f"--dataset {args.dataset} --annotator {args.annotator}` "
            f"to monitor the batches and download their results."
        )
    else:
        assert len(data.annotations) == len(data.user_prompts)
        save_out = [
            {
                "system_prompt": data.system_prompts[idx],
                "user_prompt": data.user_prompts[idx],
                "annotation_original": data.annotations_original[idx],
                "annotation": data.annotations[idx],
            }
            for idx in range(len(data.annotations))
        ]
        reannotated_file = os.path.join(save_sub_dir, "reannotated.json")
        logging.info(f"Saved final reannotated dataset to {reannotated_file}")
        with open(reannotated_file, "w") as f:
            json.dump(save_out, f, indent=4)

    # Save config yaml
    save_yaml = {
        "uuid": str(uuid.uuid4()),
        "creation_date": datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        "params": args.__dict__,
    }
    with open(f"{args.save_dir}/{save_name}/config.yaml", "w") as f:
        yaml.dump(save_yaml, f)


if __name__ == "__main__":
    main()
