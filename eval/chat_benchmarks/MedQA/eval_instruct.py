import json
import logging
from typing import Any, Dict, List, Optional

import lm_eval.models
import numpy as np
from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

# MedQA dataset with multiple-choice questions
PROMPT = """You are an expert in answering medical exam questions. Your response should be numeric (a single digit).
Medical question: {question}
Choices:
{options}
Respond with the correct choice number from 1, 2, 3, or 4:"""


class MedQABenchmark(BaseBenchmark):
    """
    MedQA Benchmark for evaluating the medical question-answering ability of LLMs.
    Loads data from Hugging Face’s bigbio/med_qa dataset.
    """

    def __init__(
        self,
        dataset_name: str = "bigbio/med_qa",
        dataset_language: str = "med_qa_en_4options_source",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MedQA benchmark.

        Args:
            dataset_name: Hugging Face dataset name (default is "bigbio/med_qa")
            dataset_language: Specific MedQA version with 4 multiple-choice options
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.dataset_name = dataset_name
        self.dataset_language = dataset_language
        self.debug = debug
        self.max_new_tokens = 32  # Limit tokens for answer generation
        self.seed = seed
        self.n_repeat = 5
        self.options_map = {"A": 1, "B": 2, "C": 3, "D": 4}

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate model responses for MedQA.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses.
        """
        examples = self.load_questions()
        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(example["options"])])
                messages = [
                    {"role": "user", "content": PROMPT.format(question=example["question"], options=options_str)}
                ]

                templated_messages = model.apply_chat_template(messages)

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
                                "seed": seed,
                            },
                        ),
                        idx,
                    )
                )

            self.logger.info("Generating responses for MedQA...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(o) for o in outputs]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model responses for MedQA using accuracy."""

        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        all_results = []
        for i in range(self.n_repeat):
            solved = sum([is_equiv(example["answer_idx"], example["model_answers"][i]) for example in examples])
            all_results.append(
                {
                    "repetition": i + 1,
                    "num_total": num_questions,
                    "num_solved": solved,
                    "accuracy": solved / num_questions,
                }
            )

        solved_avg = np.mean([result["num_solved"] for result in all_results])
        accuracy_avg = np.mean([result["accuracy"] for result in all_results])
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
        """Load MedQA questions from Hugging Face dataset."""
        dataset = load_dataset(self.dataset_name, name=self.dataset_language)

        questions = [
            {
                "question": item["question"],
                "options": [opt["value"] for opt in item["options"]],
                "answer_idx": str(self.options_map[item["answer_idx"]]),
            }
            for item in dataset["test"]
        ]

        if self.debug:
            questions = questions[:2]

        self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
        return questions

    def extract_answer(self, output: str) -> str:
        """
        Extracts the final answer from model output.
        Assumes the model returns a number corresponding to the selected option.

        Args:
            output (str): Model-generated answer

        Returns:
            str: Extracted answer or empty string if extraction fails.
        """
        try:
            answer = output.strip()
            return answer if answer.isdigit() else ""
        except:
            return ""
