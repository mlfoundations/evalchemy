import argparse
import json
import logging
import os
import uuid
from datetime import datetime

import requests
import yaml

from dcft.dataset.annotators import (
    ANNOTATOR_MAP,
    AnnotatorConfig,
    get_annotator,
    is_gpt_annotator,
)
from dcft.dataset.generation import GenerationConfig
from dcft.dataset.hf import get_dataclass_from_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_rate_limits(annotator):
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


def regenerate_dataset(args):
    # Get rate limits
    if is_gpt_annotator(args.annotator) and (
        args.max_requests_per_minute is None or args.max_tokens_per_minute is None
    ):
        max_requests, max_tokens = get_rate_limits(args.annotator)
        args.max_requests_per_minute = (
            max_requests if args.max_requests_per_minute is None else args.max_requests_per_minute
        )
        args.max_tokens_per_minute = max_tokens if args.max_tokens_per_minute is None else args.max_tokens_per_minute

    print(f"Setting max_requests_per_minute to {args.max_requests_per_minute}")
    print(f"Setting max_tokens_per_minute to {args.max_tokens_per_minute}")

    # Load data
    data = get_dataclass_from_path(args.dataset)
    assert len(data.system_prompts) == len(data.user_prompts)
    logging.info(f"Reannotating {len(data.system_prompts)} samples")

    # Do generation
    annotator_config = AnnotatorConfig(args)
    generation_config = GenerationConfig(args)
    annotator = get_annotator(args.annotator, annotator_config)
    annotator.annotate(data, generation_config)

    # Save outputs
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = f"{args.dataset.replace('/', '_')}_{args.annotator}"
    os.makedirs(f"{args.save_dir}/{save_name}", exist_ok=True)
    if args.batch:
        assert data.batch_objects is not None
        batch_objects_file = f"{args.save_dir}/{save_name}/batch_objects.json"
        with open(batch_objects_file, "w") as f:
            json.dump([obj.model_dump() for obj in data.batch_objects], f, indent=4)
        logging.info(f"Batch objects saved to {batch_objects_file}")
        logging.info(
            f"Run `python dcft/dataset/watch_gpt_batch.py --batch_objects_file {batch_objects_file} "
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
        with open(f"{args.save_dir}/{save_name}/reannotated.json", "w") as f:
            json.dump(save_out, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--annotator", type=str, default="gpt-4o-mini", choices=list(ANNOTATOR_MAP.keys()))
    parser.add_argument("--dataset", type=str, required=True, help="")
    parser.add_argument("--save_dir", type=str, default="datasets/reannotated")
    parser.add_argument("--resume", action="store_true", help="Resume from a previous run")
    parser.add_argument(
        "--batch", action="store_true", help="Whether to run in batch mode, available for GPT API annotator only"
    )

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--stop", type=str, default=None, help="parsed with .split(',')")

    # GPT API annotator parameters
    # More details at https://platform.openai.com/docs/api-reference/chat/create
    parser.add_argument("--frequency_penalty", type=float, default=0)
    parser.add_argument("--logit_bias", type=str, default=None)
    parser.add_argument("--logprobs", type=bool, default=False)
    parser.add_argument("--top_logprobs", type=int, default=None)
    parser.add_argument("--n", type=int, default=1, help="how many completions to generate for each input")
    parser.add_argument("--presence_penalty", type=float, default=0)
    parser.add_argument("--max_requests_per_minute", type=int, default=None)
    parser.add_argument("--max_tokens_per_minute", type=int, default=None)

    args = parser.parse_args()
    regenerate_dataset(args)

    save_name = f"{args.dataset.replace('/', '_')}_{args.annotator}"
    save_yaml = {
        "uuid": str(uuid.uuid4()),
        "creation_date": datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        "params": args.__dict__,
    }
    with open(f"{args.save_dir}/{save_name}/config.yaml", "w") as f:
        yaml.dump(save_yaml, f)


if __name__ == "__main__":
    main()
