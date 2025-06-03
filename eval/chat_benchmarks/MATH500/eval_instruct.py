import json
import logging
from typing import Any, Dict, List, Optional

import lm_eval.models
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class MATH500Benchmark(BaseBenchmark):
    """
    MATH500 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/MATH500/data/math500.jsonl",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MATH500 benchmark.

        Args:
            data_file: File containing the MATH500 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens

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
        all_instances = []
        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        else:
            model_name = model.model_args["model"]
        for idx, example in enumerate(examples):
            messages = [
                {"role": "user", "content": PROMPT.format(problem=example["problem"])},
            ]

            templated_messages = self._prepare_messages(messages, model)

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": self.seed,
                        },
                    ),
                    idx,
                )
            )

        # Generate model responses
        self.logger.info("Generating responses for MATH500...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output)

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        solved = sum(is_equiv(str(example["answer"]), example["model_answer"]) for example in examples)

        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "accuracy": solved / total,
            }
        )

        return results

    def load_questions(self) -> List[Dict[str, str]]:
        """Load MATH500 questions from the data file."""
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
