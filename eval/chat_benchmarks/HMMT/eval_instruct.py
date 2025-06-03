import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from datasets import load_dataset
from eval.task import BaseBenchmark

from matharena.parser import extract_answer, parse_answer, check_answers, WarningType
from matharena.possible_issues import check_number_proximity_any_order, check_all_numbers, check_output_length

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
PROMPT = """Problem: {problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"""


class HMMTBenchmark(BaseBenchmark):
    """
    HMMT Benchmark for evaluating the math reasoning of LLMs.
    https://huggingface.co/datasets/MathArena/hmmt_feb_2025

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        dataset_name: str = "MathArena/hmmt_feb_2025",
        debug: bool = False,
        max_tokens: int = 32768,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize HMMT benchmark.

        Args:
            dataset_name: Dataset containing the HMMT dataset (id, answer)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = 10

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.load_questions()
        # Prepare instances for model
        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                messages = [
                    {"role": "user", "content": PROMPT.format(problem=example["problem"])},
                ]

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
                    "problem_id": str(example["id"]) if "id" in example else str(idx),
                    "expected_answer": str(example["answer"]),
                    "reference_solution": str(example["solution"]) if "solution" in example else "",
                }

                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for HMMT...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)
        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            list_answer = "," in str(example["answer"])
            example["model_answers"] = [extract_answer(o, False, True, list_answer)[0] for o in outputs]
            example["label"] = []
        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = 0
            for example in examples:
                gold_answer, _ = parse_answer(str(example["answer"]))
                model_answer = example["model_answers"][i]
                is_correct = check_answers(model_answer, gold_answer)
                example["label"].append(is_correct)
                solved += is_correct
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

    def load_questions(self) -> List[Dict[str, str]]:
        """Load HMMT questions from the data file."""
        dataset = load_dataset(self.dataset_name, split="train")
        questions = [dict(example) for example in dataset]
        return questions
