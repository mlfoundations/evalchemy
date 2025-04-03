import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import lm_eval.models
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

# Modified version of hendrycks_math with additional instruction to mark the solution with \boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class MATH500x2Benchmark(BaseBenchmark):
    """
    MATH500x2 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

    Follows the evaluation logic of hendrycks_math answer extraction and supports multiple repetitions.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/MATH500x2/data/math500.jsonl",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MATH500 benchmark.

        Args:
            data_file: File containing the MATH500 dataset (id, problem, reference_solution, answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = 32768  # set higher to avoid truncation for reasoning models
        self.n_repeat = 2

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses or None for non-primary ranks
        """
        examples = self.load_questions()
        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            # Adjust seed per repetition
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
                # Add repetition info and metadata
                instance.repeat_idx = i
                instance.metadata = {
                    "problem_id": str(example["id"]) if "id" in example else str(idx),
                    "expected_answer": str(example["answer"]),
                    "reference_solution": str(example["reference_solution"]) if "reference_solution" in example else "",
                }
                all_instances.append(instance)

            self.logger.info("Generating responses for MATH500 (repetition %d/%d)...", i + 1, self.n_repeat)
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        # Only primary rank returns outputs
        if model.rank != 0:
            return None

        # Zip outputs so that each example has outputs for each repetition.
        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(o) for o in outputs]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions with repetition statistics."""

        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)
        all_results = []

        for i in range(self.n_repeat):
            solved = sum(
                is_equiv(str(example["answer"]), str(example["model_answers"][i]))
                for example in examples
            )
            all_results.append({
                "repetition": i + 1,
                "num_total": num_questions,
                "num_solved": solved,
                "accuracy": solved / num_questions,
            })

        solved_avg = np.mean([result["num_solved"] for result in all_results])
        accuracy_avg = np.mean([result["accuracy"] for result in all_results])
        accuracy_std = np.std([result["accuracy"] for result in all_results])
        accuracy_std_err = accuracy_std / np.sqrt(self.n_repeat)

        results.update({
            "num_total": num_questions,
            "solved_avg": solved_avg,
            "run_stats": all_results,
            "accuracy_avg": accuracy_avg,
            "accuracy_std_err": accuracy_std_err,
            "num_repeat": self.n_repeat,
        })

        return results

    def load_questions(self) -> List[Dict[str, str]]:
        """Load MATH500 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution, expected to be in the format of \\boxed{{answer}}.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found in \\boxed.
        """
        try:
            answer = remove_boxed(last_boxed_only_string(output))
            return answer
        except Exception:
            return ""
