import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class AIME24Benchmark(BaseBenchmark):
    """
    AIME24 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/zwhe99/aime24

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/AIME24/data/aime24.json",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize AIME24 benchmark.

        Args:
            data_file: File containing the AIME24 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
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
                    "expected_answer": str(example["expected_answer"]),
                    "reference_solution": str(example["reference_solution"]) if "reference_solution" in example else "",
                }

                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for AIME24...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)
        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(o) for o in outputs]

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
            solved = sum(
                [is_equiv(str(example["expected_answer"]), str(example["model_answers"][i])) for example in examples]
            )
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
        """Load AIME24 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

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
