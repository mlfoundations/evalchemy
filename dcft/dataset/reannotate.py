import os
import json
import yaml
import argparse
import uuid
from datetime import datetime
from dcft.dataset.hf import get_dataclass_from_path
from dcft.dataset.annotators import get_annotator, ANNOTATOR_MAP, AnnotatorConfig
from dcft.dataset.generation import GenerationConfig


def regenerate_dataset(args):
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
    save_out = [{
                "system_prompt": data.system_prompts[idx],
                "user_prompt": data.user_prompts[idx],
                "annotation_original": data.annotations_original[idx],
                "annotation": data.annotations[idx]
            } for idx in range(len(data.annotations))]
    
    save_name = f"{args.dataset.replace('/', '_')}_{args.annotator}"
    with open(f"{args.save_dir}/{save_name}.json", 'w') as f:
        json.dump(save_out, f, indent=4)
        

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--annotator", type=str, default="gpt-4o-2024-08-06", choices=list(ANNOTATOR_MAP.keys()))
    parser.add_argument("--dataset", type=str, required=True, help="")
    parser.add_argument("--save_dir", type=str, default="datasets/reannotated")
    parser.add_argument("--resume", action="store_true", help="Resume from a previous run")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--stop", type=str, default=None, help="parsed with .split(',')")

    # GPT API annotator parameters
    # More details at https://platform.openai.com/docs/api-reference/chat/create
    parser.add_argument("--max_requests_per_minute", type=int, default=1500)
    parser.add_argument("--max_tokens_per_minute", type=int, default=6250000)
    parser.add_argument("--frequency_penalty", type=float, default=0)
    parser.add_argument("--logit_bias", type=str, default=None)
    parser.add_argument("--logprobs", type=bool, default=False)
    parser.add_argument("--top_logprobs", type=int, default=None)
    parser.add_argument("--n", type=int, default=1, help="how many completions to generate for each input")
    parser.add_argument("--presence_penalty", type=float, default=0)

    args = parser.parse_args()
    regenerate_dataset(args)

    save_name = f"{args.dataset.replace('/', '_')}_{args.annotator}"
    save_yaml = {
        "uuid": str(uuid.uuid4()),
        "creation_date": datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        "params": args.__dict__
    }
    with open(f'{args.save_dir}/{save_name}.yml', 'w') as f:
        yaml.dump(save_yaml, f)

if __name__ == "__main__":
    main()