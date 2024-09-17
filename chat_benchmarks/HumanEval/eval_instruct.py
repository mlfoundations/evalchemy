import argparse
import json
import os
import torch
import tempfile
from pathlib import Path
from tqdm import tqdm
from lm_eval.api.instance import Instance

data_abs_dir = Path(__file__).parent / "data"

from utils.utils import extract_generation_code, languge_settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness


def build_deepseekcoder_instruction(language: str, question: str):
    return """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip().format(
        language.lower(), question.strip()
    )

def eval_instruct(model):
    results = {}
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name

    for lang in ["python", "sh"]:
        problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")

        examples = [json.loads(x) for x in open(problem_file) if x.strip()]
        print("Read {} examples for evaluation over.".format(len(examples)))

        generated_examples = []
        all_instances = []
        for idx, example in enumerate(tqdm(examples, desc="Generating")):
            prompt = build_deepseekcoder_instruction(languge_settings[lang]["full_name"], example["prompt"])
            inputs = model.apply_chat_template([{"role": "user", "content": prompt}])
            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        inputs,
                        {
                            "max_new_tokens": 1024,
                            "do_sample": False,
                        },
                    ),
                    idx,
                )
            )

        outputs = model.generate_until(all_instances)

        for idx, example in enumerate(examples):
            example["output"] = outputs[idx]

        generated_examples = [extract_generation_code(example, lang_code=lang) for example in examples]

        results[lang] = generated_examples
        temp_file_path = os.path.join(temp_dir, f"generated_{lang}.jsonl")
        with open(temp_file_path, "w", encoding="utf-8") as fw:
            for ex in generated_examples:
                fw.write(json.dumps(ex) + "\n")
        print("Save {} processed examples into temporary file over!".format(len(generated_examples)))

    print("Generate all over!!!")
    results["temp_dir_obj"] = temp_dir_obj
    return results


def evaluate(results):
    temp_dir_obj = results["temp_dir_obj"]
    temp_dir = temp_dir_obj.name

    evaluation_results = {}
    for lang in ["python", "sh"]:
        problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")
        temp_file_path = os.path.join(temp_dir, f"generated_{lang}.jsonl")

        result = evaluate_functional_correctness(
            input_file=temp_file_path,
            tmp_dir=temp_dir,
            n_workers=8,
            timeout=3.0,
            problem_file=problem_file,
            language=lang,
        )
        evaluation_results[lang] = result
    temp_dir_obj.cleanup()
    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name or path")
    parser.add_argument("--output_path", type=str, help="output path of your generation")
    parser.add_argument("--language", type=str, help="langauge")
    parser.add_argument("--temp_dir", type=str, help="temp dir for evaluation", default="tmp")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
    pass
