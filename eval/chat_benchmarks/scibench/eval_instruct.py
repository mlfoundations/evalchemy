import json
import logging
import math
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark
from openai import OpenAI
import time

# SciBench's identical prompt
sys_cal_box2 = """
Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating "The answer is therefore \\boxed{[ANSWER]}."
"""


def remove_not(x):
    match_number = re.compile("[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?")
    result = re.findall(match_number, x)
    if len(result) != 0:
        return re.split(match_number, x)[-1]
    return None


def parse_not(inputs):
    try:
        if not inputs:
            return "", ""
        if "\times" in inputs:
            x, ab = inputs.split("\times")
        elif "\\times" in inputs:
            x, ab = inputs.split("\\times")
        elif "*" in inputs:
            x, ab = inputs.split("*")
        else:
            return inputs
        return x, ab
    except:
        return "", ""


def cal_not(inputs):
    try:
        x, ab = list(inputs)
        match_number = re.compile("10\^[{]?\ *-?[0-9]+\ *[}]?")
        ab = re.findall(match_number, ab)[0]
        ab = ab[ab.find("^") + 1 :]
        if "{" in ab:
            ab = ab[ab.find("{") + 1 :]
        if "}" in ab:
            ab = ab[: ab.find("}")]
        x = x.strip()
        out = float(x) * 10 ** float(ab)
        return str(out)
    except:
        print("error")
    return inputs


def remove_boxed(s):
    left = "oxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        answer = s[len(left) : -1]
        if "=" in answer:
            answer = answer.split("=")[-1].lstrip(" ")
        return answer
    except:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("oxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    return retval


def parse_math_answer(raw_string):
    return remove_boxed(last_boxed_only_string(raw_string))


def equiv(model_output, answer, unit):
    """SciBench's exact equiv function"""
    model_output = model_output.replace(",", "")
    try:
        ans = float(answer.strip())
        first = math.isclose(float(model_output.strip()), ans, rel_tol=0.05)
    except:
        first = False
    try:
        model = model_output.strip().split()[0]
        second = math.isclose(float(model.strip()), ans, rel_tol=0.05)
    except:
        second = False
    if first or second:
        return True
    return False


@dataclass
class SciBenchConfig:
    """Configuration for SciBench evaluation."""

    categories: List[str] = field(default_factory=lambda: ["chemmc"])
    temperature: float = 0.0
    max_new_tokens: int = 1024
    do_sample: bool = False


class SciBenchBenchmark(BaseBenchmark):
    """SciBench benchmark implementation."""

    def __init__(
        self,
        categories: List[str] = None,
        config: Optional[SciBenchConfig] = None,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)
        self.config = config or SciBenchConfig()
        if categories:
            self.config.categories = categories
        self.debug = debug

    def _load_dataset(self, category: str):
        """Load dataset from JSON files in the specified data directory"""
        try:
            data_dir = Path("./data")  # TODO: CHANGE TO DATA DIRECTORY IF NEEDED
            file_path = data_dir / f"{category}.json"

            with open(file_path, "r") as f:
                dataset = json.load(f)

            # Filter problems for the specific category
            problems = [
                {
                    "problem_text": item["problem_text"],
                    "answer_number": item["answer_number"],
                    "unit": item["unit"],
                    "original_unit": item["unit"],  # Using same unit as original since dataset doesn't distinguish
                    "source": item["source"],
                }
                for item in dataset
                if item["source"] == category
            ]

            # Process units if needed
            processed_problems = []
            for problem_data in problems:
                unit = problem_data["unit"]
                base_unit = remove_not(unit)
                if base_unit:
                    unit = base_unit
                problem_data["unit"] = unit
                processed_problems.append(problem_data)

            self.logger.info(f"Loaded {len(processed_problems)} problems for category {category}")
            return processed_problems

        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def call_engine(self, messages, temperature=0, n=1, patience=100000, sleep_time=0):
        """Match eval_zero.py's implementation for API calls using new OpenAI API"""
        client = OpenAI()  # This will automatically use OPENAI_API_KEY from env

        while patience > 0:
            patience -= 1
            try:
                response = client.chat.completions.create(
                    model="gpt-4", messages=messages, temperature=temperature, n=n  # or use self.config.model
                )
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction != "" and prediction is not None:
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction[0] != "" and prediction[0] is not None:
                        return prediction
            except Exception as e:
                self.logger.error(f"OpenAI API error: {e}")
                if sleep_time > 0:
                    import time

                    time.sleep(sleep_time)
        return ""

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate responses for all problems using OpenAI API."""
        results = {}

        try:
            category = "chemmc"
            problems = self._load_dataset(category)
            print(f"\nProcessing category {category}")
            print(f"Total problems loaded: {len(problems)}")

            if self.debug:
                problems = problems[: min(10, len(problems))]
                print(f"Debug mode: Using {len(problems)} problems")

            ids = [f"{category}_{i}" for i in range(len(problems))]
            metadata = {
                "answer_number": [],
                "unit": [],
                "original_unit": [],
            }
            outputs = []

            for i, problem in enumerate(problems):
                unit_prob = problem["unit"]
                problem_text = f"{problem['problem_text']} The unit of the answer is {unit_prob}."

                messages = [
                    {"role": "system", "content": sys_cal_box2},
                    {"role": "user", "content": f"Q: {problem_text}\nA: The answer is"},
                ]

                print(f"\nProblem {i}:")
                print(f"Text: {problem_text}")
                output = self.call_engine(
                    messages, temperature=self.config.temperature, n=1, patience=100000, sleep_time=1
                )
                print(f"Response: {output}")

                outputs.append(output)
                metadata["answer_number"].append(problem["answer_number"])
                metadata["unit"].append(problem["unit"])
                metadata["original_unit"].append(problem["original_unit"])

            results[category] = {
                "ids": ids,
                "outputs": outputs,
                "metadata": metadata,
            }
            print(f"\nFinal results structure:")
            print(f"Category: {category}")
            print(f"Number of responses: {len(outputs)}")
            print(f"Sample response: {outputs[0] if outputs else 'No outputs'}")

        except Exception as e:
            print(f"\nError processing category {category}: {e}")

        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate all responses using SciBench's exact evaluation logic."""
        eval_results = {}

        category = "chemmc"
        category_results = results.get(category, {})
        correct = 0
        total = 0

        for output, answer, unit, original_unit in zip(
            category_results.get("outputs", []),
            category_results.get("metadata", {}).get("answer_number", []),
            category_results.get("metadata", {}).get("unit", []),
            category_results.get("metadata", {}).get("original_unit", []),
        ):
            model_output = parse_math_answer(output)
            if not model_output:
                numbers = re.findall(r"\\boxed{([^}]*)}", output)
                if numbers:
                    model_output = numbers[-1].strip()
                else:
                    total += 1
                    continue

            if unit != original_unit:
                notation = parse_not(model_output)
                if notation and len(notation) == 2:
                    model_output = cal_not(notation)
                    answer = cal_not((answer, original_unit))

            if isinstance(model_output, tuple):
                model_output = model_output[0]

            if equiv(str(model_output), answer, unit):
                correct += 1
            total += 1

        eval_results[category] = (correct / total) * 100 if total > 0 else 0.0

        return eval_results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate all responses using SciBench's exact evaluation logic."""
        eval_results = {}
        total_score = 0
        num_categories = 0

        category = "chemmc"
        category_results = results.get(category, {})
        correct = 0
        total = 0

        for output, answer, unit, original_unit in zip(
            category_results.get("outputs", []),
            category_results.get("metadata", {}).get("answer_number", []),
            category_results.get("metadata", {}).get("unit", []),
            category_results.get("metadata", {}).get("original_unit", []),
        ):
            model_output = parse_math_answer(output)
            if not model_output:
                numbers = re.findall(r"\\boxed{([^}]*)}", output)
                if numbers:
                    model_output = numbers[-1].strip()
                else:
                    total += 1
                    continue

            if unit != original_unit:
                notation = parse_not(model_output)
                if notation and len(notation) == 2:
                    model_output = cal_not(notation)
                    answer = cal_not((answer, original_unit))

            if isinstance(model_output, tuple):
                model_output = model_output[0]

            if equiv(str(model_output), answer, unit):
                correct += 1
            total += 1

        category_score = (correct / total) * 100 if total > 0 else 0.0
        eval_results[category] = category_score
        total_score += category_score
        num_categories += 1

        # Add average score
        eval_results["average"] = total_score / num_categories if num_categories > 0 else 0.0

        return eval_results
