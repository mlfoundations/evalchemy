import logging
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

########################################################################

# Adapted from https://github.com/dair-iitd/jeebench/blob/main/inference.py


prompt_library = {
    "MCQ": "In this problem, only one option will be correct. Give a detailed solution and end the solution with the final answer.",
    "MCQ(multiple)": "In this problem, multiple options can be correct. Give a detailed solution and end the solution with the final answer.",
    "Integer": "In this problem, the final answer will be a non-negative integer. Give a detailed solution and end the solution with the final answer.",
    "Numeric": "In this problem, the final will be a numeric value. Give the numerical answer correct upto the 2nd decimal digit. Give a detailed solution and end the solution with the final answer.",
}


def format_message(question):
    prefix_prompt = prompt_library[question["type"]]
    suffix_prompt = ""

    stripped_question = question.replace("\n\n", "\n").strip()

    prompt = prefix_prompt + "\n\n" + "Problem: " + stripped_question + suffix_prompt

    content = prompt.strip()
    messages = [{"role": "user", "content": content}]

    return messages


########################################################################


def extract_answer(output: str) -> str:
    raise NotImplementedError


class JEEBenchBenchmark(BaseBenchmark):
    """
    JEEBench, comprising "515 challenging preengineering mathematics, physics and chemistry problems from the highly competitive IIT JEE-Advanced exam."
    Link: https://huggingface.co/datasets/daman1209arora/jeebench
    """

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize JEEBench benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = 32768  # set higher to avoid truncation for reasoning models
        self.seed = seed
        self.n_repeat = 3

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
            examples = examples.select(range(2))
            self.logger.info(f"Debug mode: using 2 examples")

        # Prepare instances for model
        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                messages = format_message(example)

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
            example["model_answers"] = [extract_answer(o) for o in outputs]
            examples_list.append(example)

        return {"examples": examples_list}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        if results is None:
            return None

        # TODO

        return results

    def load_questions(self) -> Dataset:
        """
        Load JEEBench questions from source.
        """
        self.logger.info("Loading JEEBench questions from source...")
        dataset = load_dataset("daman1209arora/jeebench", split="test")
        self.logger.info(f"{len(dataset)} examples retrieved.")
        return dataset
