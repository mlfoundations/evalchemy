"""
This script will read the results from result_dirs_parsed and summarize the results.
We want to summarize the model performance on each example in each task that is evaluated.
We first save the model names that are correct and incorrect on each example, and then
we can compute the ratio on each example to know how many models are correct on each example.
We also include the reasoning provided by different models for each example.
"""

import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.eval_utils import model_name_replacement

# example_data_structure from result_dirs_parsed/gsm/Athene-70B.json
"""
[
  {
    "session_id": "gsm8k-main-test-#0",
    "chat_history": [
      "\n## Question: \n\nJanet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n\n\n## Instruction \n\nPlease answer this question by first reasoning and then providing your answer.\nPresent your reasoning and solution in the following json format. \nPlease show your final answer in the `answer` field, e.g.,`\"answer\": \"42\"`.\n\n```json\n{\n    \"reasoning\": \"___\",\n    \"answer\": \"___\"\n}\n```\n"
    ],
    "model_input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n## Question: \n\nJanet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n\n\n## Instruction \n\nPlease answer this question by first reasoning and then providing your answer.\nPresent your reasoning and solution in the following json format. \nPlease show your final answer in the `answer` field, e.g.,`\"answer\": \"42\"`.\n\n```json\n{\n    \"reasoning\": \"___\",\n    \"answer\": \"___\"\n}\n```<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "output": [
      "{\n  \"reasoning\": \"Janet starts with 16 eggs per day. She uses 3 for breakfast and 4 for muffins, leaving 16 - 3 - 4 = 9 eggs to sell. At $2 per egg, she earns 9 \u00d7 $2 = $18 daily.\",\n  \"answer\": \"18\"\n}"
    ],
    "generator": "Nexusflow/Athene-70B",
    "configs": {
      "engine": "vllm",
      "repetition_penalty": 1.0,
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 4096
    },
    "dataset": "gsm",
    "id": "gsm8k-main-test-#0",
    "question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "answer": "# 18",
    "source": "openai/gsm8k",
    "config": "main",
    "task_type": "qa",
    "reasoning": "Janet starts with 16 eggs per day. She uses 3 for breakfast and 4 for muffins, leaving 16 - 3 - 4 = 9 eggs to sell. At $2 per egg, she earns 9 \u00d7 $2 = $18 daily.",
    "model_answer": {
      "raw": "18",
      "sanitized": "18",
      "first_number": "18"
    },
    "correct_answer": {
      "raw": "18",
      "sanitized": "18",
      "first_number": "18"
    },
    "matched": true
  },
"""


def summarize_results(task_name):
    """
    This function will read the results from result_dirs_parsed and summarize the results.
    We want to summarize the model performance on each example in each task that is evaluated.
    We first save the model names that are correct and incorrect on each example, and then
    we can compute the ratio on each example to know how many models are correct on each example.
    We also include the reasoning provided by different models for each example.
    """
    model_names_unique = set()
    task_summary = {}
    # go through each file in the result_dirs_parsed/task_name
    for file in os.listdir(f"result_dirs_parsed/{task_name}"):
        if file.endswith(".json"):
            with open(f"result_dirs_parsed/{task_name}/{file}", "r") as f:
                data = json.load(f)
                model_name = file.replace(".json", "")
                model_name = model_name_replacement(model_name)
                model_names_unique.add(model_name)
                # go through each example in the data
                for example in data:
                    # if the example is not in the dictionary, add it
                    if example["id"] not in task_summary:
                        task_summary[example["id"]] = {}
                        if "choices" in example:
                            choices = example["choices"]
                            formatted_choices = "\n".join(
                                [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
                            )
                            task_summary[example["id"]]["question"] = example["question"] + "\n\n" + formatted_choices
                        else:
                            task_summary[example["id"]]["question"] = (
                                example["question"] if "question" in example else example["problem"]
                            )
                        task_summary[example["id"]]["correct_answer"] = (
                            example["correct_answer"]["sanitized"]
                            if "sanitized" in example["correct_answer"]
                            else example["correct_answer"]
                        )
                        # save the data inputs and answers
                        task_summary[example["id"]]["correct_models"] = []
                        task_summary[example["id"]]["incorrect_models"] = []
                        task_summary[example["id"]]["model_answers"] = {}
                        task_summary[example["id"]]["reasoning"] = {}
                    # Check if matched and not "No answer extracted"
                    if (
                        example.get("matched", False) and example.get("matched") != "No answer extracted"
                    ) or example.get("solved", False):
                        task_summary[example["id"]]["correct_models"].append(model_name)
                    else:
                        task_summary[example["id"]]["incorrect_models"].append(model_name)
                    # save the model answer
                    task_summary[example["id"]]["model_answers"][model_name] = (
                        example.get("model_answer", {}).get("sanitized", "")
                        if "sanitized" in example.get("model_answer", {})
                        else example.get("model_answer", "")
                    )
                    # save the model reasoning
                    task_summary[example["id"]]["reasoning"][model_name] = example.get("reasoning", "")

    # now we have a dictionary of all the examples and the models that are correct and incorrect on each example
    # we want to compute the ratio of correct models to incorrect models for each example
    for example in task_summary:
        task_summary[example]["correct_ratio"] = len(task_summary[example]["correct_models"]) / (
            len(task_summary[example]["correct_models"]) + len(task_summary[example]["incorrect_models"])
        )
        # convert the model lists to string
        task_summary[example]["correct_models"] = "|".join(task_summary[example]["correct_models"])
        task_summary[example]["incorrect_models"] = "|".join(task_summary[example]["incorrect_models"])
    # rank the examples by the correct_ratio from smallest to largest
    task_summary = dict(sorted(task_summary.items(), key=lambda x: x[1]["correct_ratio"]))
    # now we want to save the task_summary to a file
    with open(f"state_of_limit/task_summary_{task_name}.json", "w") as f:
        json.dump(task_summary, f, indent=2)
        print(f"Saved {len(model_names_unique)} unique models to {f.name}")


# test with gsm
summarize_results("gsm")
summarize_results("math-l5")
summarize_results("crux")
summarize_results("mmlu-redux")
# summarize_results("zebra-grid")
