import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from utils import compute_score, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

########################################################################

# Adapted from https://github.com/dair-iitd/jeebench/blob/main/inference.py


PROMPT_LIBRARY = {
    "MCQ": "In this problem, only one option will be correct. Give a detailed solution and end the solution with the final answer.",
    "MCQ(multiple)": "In this problem, multiple options can be correct. Give a detailed solution and end the solution with the final answer.",
    "Integer": "In this problem, the final answer will be a non-negative integer. Give a detailed solution and end the solution with the final answer.",
    "Numeric": "In this problem, the final will be a numeric value. Give the numerical answer correct upto the 2nd decimal digit. Give a detailed solution and end the solution with the final answer.",
}

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for JEEBench benchmark"
    )


def format_message(question, prompt_library):
    prefix_prompt = prompt_library[question["type"]]
    suffix_prompt = ""

    stripped_question = question["question"].replace("\n\n", "\n").strip()

    prompt = prefix_prompt + "\n\n" + "Problem: " + stripped_question + suffix_prompt

    content = prompt.strip()
    messages = [{"role": "user", "content": content}]

    return messages


########################################################################


def prompt_for_boxed_answer(prompt_library):
    """
    Mofify prompt_library in-place to elicit final answer in parseable boxed format.
    """

    EXPECTED_KEYS = {"MCQ", "MCQ(multiple)", "Integer", "Numeric"}
    assert (
        set(prompt_library.keys()) == EXPECTED_KEYS
    ), f"Unexpected keys in prompt_library: {set(prompt_library.keys())}"

    prompt_library[
        "MCQ"
    ] += " Mark your solution, which should be exactly one multiple-choice letter, with \\boxed\nAnswer:"
    prompt_library[
        "MCQ(multiple)"
    ] += " Mark your solution, which should be one or more multiple-choice letter(s), with \\boxed\nAnswer:"
    prompt_library["Integer"] += " Mark your solution with \\boxed\nAnswer:"
    prompt_library["Numeric"] += " Mark your solution with \\boxed\nAnswer:"

    return prompt_library


class JEEBenchBenchmark(BaseBenchmark):
    """
    JEEBench, comprising "515 challenging preengineering mathematics, physics and chemistry problems from the highly competitive IIT JEE-Advanced exam."
    Link: https://huggingface.co/datasets/daman1209arora/jeebench
    """

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize JEEBench benchmark.

        Args:
            debug: If set, only evaluate on 3 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = 3

        self.prompt_library = prompt_for_boxed_answer(PROMPT_LIBRARY)

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """

        self.logger.info("Generating responses in 'normal' mode (no CoT, SC, or Exam mode)...")

        examples = self.load_questions()
        if self.debug:
            examples = examples.select(range(3))
            self.logger.info(f"Debug mode: using 3 examples")

        # Prepare instances for model
        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                messages = format_message(example, self.prompt_library)

                templated_messages = self._prepare_messages(messages, model)

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": seed,
                        },
                    ),
                    idx,
                )

                # Add repetition information to instance metadata
                instance.repeat_idx = i
                instance.metadata = {
                    "problem_id": str(example["index"]) if "index" in example else str(idx),
                    "expected_answer": str(example["gold"]),
                    "subject": str(example["subject"]),
                    "type": str(example["type"]),
                }

                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for JEEBench...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)
        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        examples_list = []

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(o) for o in outputs]
            examples_list.append(example)

        return {"examples": examples_list}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each example and repetition
        for example in examples:
            example["score"] = [
                compute_score(example["gold"], example["model_answers"][i], example["type"])
                for i in range(self.n_repeat)
            ]

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = sum([example["score"][i] for example in examples])
            all_results.append(
                {
                    "repetition": i + 1,
                    "num_total": num_questions,
                    "num_solved": solved,
                    "accuracy": solved / num_questions,
                }
            )

        # Calculate overall statistics
        solved_avg = np.mean([result["num_solved"] for result in all_results])
        accuracy_avg = np.mean([result["accuracy"] for result in all_results])
        accuracy_std = np.std([result["accuracy"] for result in all_results])
        accuracy_std_err = np.std([result["accuracy"] for result in all_results]) / np.sqrt(self.n_repeat)

        results.update(
            {
                "num_total": num_questions,
                "solved_avg": solved_avg,
                "run_stats": all_results,
                "accuracy_avg": accuracy_avg,
                "accuracy_std_err": accuracy_std_err,
                "num_repeat": self.n_repeat,
            }
        )

        return results

    def load_questions(self) -> Dataset:
        """
        Load JEEBench questions from source.
        """
        self.logger.info("Loading JEEBench questions from source...")
        dataset = load_dataset("daman1209arora/jeebench", split="test", cache_dir=HF_HUB_CACHE)
        self.logger.info(f"{len(dataset)} examples retrieved.")
        return dataset

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution, which is expected to be in the format of \boxed{answer}.

        Uses the same logic as hendrycks_math.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found in \boxed.
        """
        try:
            answer = remove_boxed(last_boxed_only_string(output))
            return answer
        except:
            return ""
