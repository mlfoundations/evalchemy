import logging
import os
import random
from typing import Any, Dict, List, Optional

import lm_eval.models
import numpy as np
from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

from .testing_utils import get_multiple_choice_answer

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for MMLUPro benchmark"
    )


class MMLUProBenchmark(BaseBenchmark):
    """
    MMLUPro (500 subset) Benchmark for evaluating multiple choice reasoning of LLMs.
    https://huggingface.co/datasets/mlfoundations-dev/mmlu_pro_eval_500subset
    """

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MMLUPro (500 subset) benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "mlfoundations-dev/mmlu_pro_eval_500subset"
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = 32768
        self.n_repeat = 1

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and examples
        """
        examples = self.load_questions()

        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        else:
            model_name = model.model_args["model"]

        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                messages = [
                    {"role": "user", "content": example["prompt"]},
                ]

                templated_messages = self._prepare_messages(messages, model)

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": True,
                            "temperature": 0.7,
                            "max_new_tokens": self.max_new_tokens,
                            "seed": seed,
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = i
                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for MMLUPro...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [get_multiple_choice_answer(o) for o in outputs]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = sum([example["answer"] == example["model_answers"][i] for example in examples])

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

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load MMLUPro (500 subset) questions from the dataset."""
        dataset = load_dataset(self.dataset_name, cache_dir=HF_HUB_CACHE)
        questions = [row for row in dataset["test"]]
        if self.debug:
            questions = questions[:2]
        self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
        return questions
