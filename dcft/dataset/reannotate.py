import os
import json
import argparse
from dcft.dataset.hf import get_dataclass_from_path
from dcft.dataset.annotators import get_annotator, ANNOTATOR_MAP


def regenerate_dataset(args):
    # Load data
    data = get_dataclass_from_path(args.dataset)
    assert len(data.system_prompts) == len(data.user_prompts)

    # Do generation
    generation_args = {
        "temperature": args.temperature
    }
    annotator = get_annotator(args.annotator, **generation_args)
    annotator.annotate(data)

    # Save outputs
    assert len(data.annotations) == len(data.user_prompts)
    os.makedirs(args.save_dir, exist_ok=True)
    save_out = [{
                "system_prompt": data.system_prompts[idx],
                "user_prompt": data.user_prompts[idx],
                "annotation_gtruth": data.annotations_gtruth[idx],
                "annotation": data.annotations[idx]
            } for idx in range(len(data.annotations))]
    with open(f"{args.save_dir}/{args.dataset.replace('/', '_')}.json", 'w') as f:
        json.dump(save_out, f, indent=4)
        

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--annotator", type=str, default="gpt-4o-mini", choices=list(ANNOTATOR_MAP.keys()))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dataset", type=str, required=True, help="")
    parser.add_argument("--save_dir", type=str, default="datasets/reannotated")

    args = parser.parse_args()
    regenerate_dataset(args)


if __name__ == "__main__":
    main()