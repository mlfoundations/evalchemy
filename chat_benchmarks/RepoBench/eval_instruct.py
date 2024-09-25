import json
import os
import tempfile
from tqdm import tqdm
from typing import Dict, List

from archive_data.utils import load_data, construct_trainable_data, get_first_line_not_comment
from data.utils import construct_prompt
from datasets import load_dataset
from evaluation.metrics import exact_match_score, edit_similarity_score, codebleu_score
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM


def eval_instruct(model: HFLM) -> Dict[str, tempfile.TemporaryDirectory]:
    """
    Evaluates the given model on all RepoBench v1.1 subsets and programming languages (python and java).

    Args:
        model (HF): The model to evaluate.

    Returns:
        Dict[str, Any]: A dictionary containing the results, including a temporary directory object where the outputs are saved.
    """
    max_token_nums = 2000

    # Create the save directory
    results = {}
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name

    for lang in ["python", "java"]:
        datasets = load_dataset(f'tianyang/repobench_{lang}_v1.1', verification_mode="no_checks")
        for subset, dataset in datasets.items():
            generated_examples = []
            all_instances = []

            for idx, example in enumerate(tqdm(dataset, desc="Generating", total=len(dataset))):
                prompt = construct_prompt(example, tokenizer=model.tokenizer, max_token_nums=max_token_nums, language=lang)

                all_instances.append(
                    Instance(
                        "generate_until",
                        example,
                        (
                            prompt,
                            {"max_new_tokens": 64, "temperature": 0.2, "top_p": 0.95, "do_sample": True},
                        ),
                        idx,
                    )
                )

            outputs = model.generate_until(all_instances)

            generated_examples = []
            for idx, example in enumerate(tqdm(dataset, desc="Generating", total=len(dataset))):
                example["idx"] = idx
                example["gpt_completion"] = get_first_line_not_comment(outputs[idx], language=lang)
                example["label"] = example['next_line']
                generated_examples.append(example)
            print("Generate all over!!!")

            with open(f"{temp_dir}/repobench_{subset}_{lang}.jsonl", "w", encoding="utf-8") as fw:
                for ex in generated_examples:
                    fw.write(json.dumps(ex) + "\n")
                print("Save {} processed examples into {} over!".format(len(generated_examples), temp_dir))

    results["temp_dir_obj"] = temp_dir_obj
    return results


def eval_instruct_legacy(model: HFLM) -> Dict[str, tempfile.TemporaryDirectory]:
    """
    Evaluates the given model on all RepoBench v0 subsets and programming languages (python and java).
    To dowload repobench v0 dataset, follow these steps:
        gdown --id '1HvFFnOybTKEJCrEypWh4ftmW6DZBaiK_' --output ./archive_data/test.zip
        unzip ./archive_data/test.zip -d ./archive_data/
        rm ./archive_data/test.zip

    Args:
        model (HF): The model to evaluate.

    Returns:
        Dict[str, Any]: A dictionary containing the results, including a temporary directory object where the outputs are saved.
    """
    prefix_token = "<fim_prefix>"
    suffix_token = "<fim_suffix><fim_middle>"

    # Create the save directory
    results = {}
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name

    for lang in ["python", "java"]:
        for subset in ["cross_file_first", "cross_file_random", "in_file"]:

            dataset = load_data(split="test", task="completion", language=lang, length="2k", setting=subset)

            examples = construct_trainable_data(dataset, language=lang)
            print("Read {} examples for evaluation over.".format(len(examples)))

            generated_examples = []
            all_instances = []
            for idx, example in enumerate(tqdm(examples, desc="Generating", total=len(examples))):

                prompt = example["data"]
                label = example["label"]

                if "star" in model._model.config._name_or_path:  # starcoder
                    prompt = prefix_token + prompt + suffix_token

                tokenizer = model.tokenizer
                # input_prompt = tokenizer(prompt, return_tensors="pt", padding=True)
                # if len(input_prompt) > 2048 - 64: # context is too long
                # continue

                all_instances.append(
                    Instance(
                        "generate_until",
                        example,
                        (
                            prompt,
                            {"max_new_tokens": 64, "temperature": 0.2, "top_p": 0.95, "do_sample": True},
                        ),
                        idx,
                    )
                )

            outputs = model.generate_until(all_instances)

            generated_examples = []
            for idx, example in enumerate(tqdm(examples, desc="Generating")):
                example["idx"] = idx
                example["gpt_completion"] = get_first_line_not_comment(outputs[idx], language=lang)
                generated_examples.append(example)
            print("Generate all over!!!")

            with open(f"{temp_dir}/repobench_{subset}_{lang}.jsonl", "w", encoding="utf-8") as fw:
                for ex in generated_examples:
                    fw.write(json.dumps(ex) + "\n")
                print("Save {} processed examples into {} over!".format(len(generated_examples), temp_dir))

    results["temp_dir_obj"] = temp_dir_obj
    return results


def evaluate(results: Dict[str, tempfile.TemporaryDirectory]) -> List[str]:
    """
    Evaluates the results from the generated outputs.

    Args:
        results (Dict[str, Any]): A dictionary containing the temporary directory where the generated outputs are saved.

    Returns:
        List[str]: A list of evaluation results, including metrics like exact match (EM) and edit similarity (ES) scores.
    """

    temp_dir_obj = results["temp_dir_obj"]
    temp_dir = temp_dir_obj.name

    total_data_points = 0
    total_em_model, total_es_model, total_cb_model = 0, 0, 0

    results = []
    for lang in ["python", "java"]:
        for subset in ["cross_file_first", "cross_file_random", "in_file"]:

            filepath = os.path.join(temp_dir, f"repobench_{subset}_{lang}.jsonl")
            seen_indices = set()  # Track seen indices for the current subset

            # check if the file exists
            if not os.path.exists(filepath):
                print(f"{filepath} not found for the model")
                continue

            with open(filepath, "r") as f:
                data = []
                for line in f:
                    entry = json.loads(line.strip())
                    data.append(entry)

                if len(data) == 0:
                    continue

                ground_truth = [d["label"] for d in data]
                generated = [d["gpt_completion"] for d in data]

                em_model = round(exact_match_score(ground_truth, generated) * 100, 2)
                es_model = round(edit_similarity_score(ground_truth, generated), 2)

                # accumulate the data points and the metrics
                total_data_points += len(data)
                total_em_model += em_model * len(data)
                total_es_model += es_model * len(data)

                print(f"Language: {lang}, Subset: {subset} with {len(data)} data points")
                results.append(f"Language: {lang}, Subset: {subset} with {len(data)} data points")
                print(f"EM: {em_model}, ES: {es_model}")

                print("-" * 30)

        # calculate the weighted averages
        if total_data_points > 0:
            avg_em_model = round(total_em_model / total_data_points, 2)
            avg_es_model = round(total_es_model / total_data_points, 2)

            print("Weighted Averages:")
            results.append(f"Weighted Averages: Language: {lang}, EM: {avg_em_model}, ES: {avg_es_model}\n")
            print(f"Language: {lang}, EM: {avg_em_model}, ES: {avg_es_model}\n")

        else:
            print("No data points were found for evaluation.")

    temp_dir_obj.cleanup()
    return results
