import argparse
import json
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
    if is_gpt_annotator(args.annotator):
        max_requests, max_tokens = get_rate_limits(args.annotator)
        args.max_requests_per_minute = max_requests
        args.max_tokens_per_minute = max_tokens

    print(f"Setting max_requests_per_minute to {args.max_requests_per_minute}")
    print(f"Setting max_tokens_per_minute to {args.max_tokens_per_minute}")

    # Load data
    data = get_dataclass_from_path(args.dataset)
    assert len(data.system_prompts) == len(data.user_prompts)
    print(f"Reannotating {len(data.system_prompts)} samples")

    # Do generation
    annotator_config = AnnotatorConfig(args)
    generation_config = GenerationConfig(args)
    annotator = get_annotator(args.annotator, annotator_config)
    annotator.annotate(data, generation_config)

    # Save outputs
    assert len(data.annotations) == len(data.user_prompts)
    os.makedirs(args.save_dir, exist_ok=True)
    save_out = [
        {
            "system_prompt": data.system_prompts[idx],
            "user_prompt": data.user_prompts[idx],
            "annotation_original": data.annotations_original[idx],
            "annotation": data.annotations[idx],
        }
        for idx in range(len(data.annotations))
    ]
    with open(f"{args.save_dir}/{args.dataset.replace('/', '_')}.json", "w") as f:
        json.dump(save_out, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--annotator", type=str, default="gpt-4o-2024-08-06", choices=list(ANNOTATOR_MAP.keys()))
    parser.add_argument("--dataset", type=str, required=True, help="")
    parser.add_argument("--save_dir", type=str, default="datasets/reannotated")

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

    args = parser.parse_args()
    regenerate_dataset(args)

    save_yaml = {
        "uuid": str(uuid.uuid4()),
        "creation_date": datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        "params": args.__dict__,
    }
    with open(f"{args.save_dir}/config.yml", "w") as f:
        yaml.dump(save_yaml, f)


if __name__ == "__main__":
    main()
