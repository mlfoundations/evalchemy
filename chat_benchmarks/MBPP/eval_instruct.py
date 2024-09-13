import argparse
import json
import os
import torch
import re
from pathlib import Path
from tqdm import tqdm
from lm_eval.api.instance import Instance

data_abs_dir = Path(__file__).parent / "data"

from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness


def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex["text"], ex["test_list"], ex["code"]
        ex_prompt = format_test_example(q, test, code)
        example_prompt = "- Example {}:\n{}".format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex["text"], ex["test_list"], ex["code"]

        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = """
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
""".strip().format(
            "\n\n".join(examples_str), prompt
        )
        yield {"task_id": ex["task_id"], "prompt": prompt_with_shots}


def convert_for_evaluation(example):
    gpt_completion = example["gpt_completion"]
    generation = gpt_completion
    try:
        code_block: str = re.findall(f"```python\n(.*?)```", gpt_completion, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    example["generation"] = generation
    return example


def eval_instruct(model, output_path, **kwargs):
    saved_path = output_path
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"mbpp.jsonl")

    examples = list(read_test_examples(problem_file))
    print("Read {} examples for evaluation over.".format(len(examples)))
    all_instances = []
    for idx, example in enumerate(tqdm(examples, desc="Generating")):
        prompt = example["prompt"]
        inputs = model.apply_chat_template([{"role": "user", "content": prompt}])
        all_instances.append(
            Instance(
                "generate_until",
                example,
                (
                    inputs,
                    {
                        "max_new_tokens": 512,
                        "do_sample": False,
                    },
                ),
                idx,
            )
        )

    outputs = model.generate_until(all_instances)

    generated_examples = []
    for idx, example in enumerate(tqdm(examples, desc="Generating")):
        example["gpt_completion"] = outputs[idx]
        generated_examples.append(example)

    genererated_examples = [convert_for_evaluation(example) for example in generated_examples]

    print("Generate all over!!!")
    with open(f"{output_path}_mbpp", "w", encoding="utf-8") as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + "\n")
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))
    results = {"saved_path": f"{output_path}_mbpp"}
    return results


def evaluate(results):
    temp_dir = "tmp"
    result = evaluate_functional_correctness(
        input_file=results["saved_path"],
        tmp_dir=temp_dir,
        problem_file=os.path.join(data_abs_dir, f"mbpp_test.jsonl"),
        language="python",
        is_mbpp=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name or path")
    parser.add_argument("--output_path", type=str, help="output path of your generation")
    parser.add_argument("--temp_dir", type=str, help="temp dir for evaluation", default="tmp")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
    pass
