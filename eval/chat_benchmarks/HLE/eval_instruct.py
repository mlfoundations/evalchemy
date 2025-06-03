import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from run_judge_results import judge_all_responses

from eval.task import BaseBenchmark

from .testing_utils import get_multiple_choice_answer

########################################################################

# Adapted from https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_model_predictions.py

SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nAnswer: {your chosen multiple choice letter}. Include only the letter, no other text."

SYSTEM_MC = "Your response should be in the following format:\nAnswer: {your chosen multiple choice letter}. Include only the letter, no other text."

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for HLE benchmark"
    )


def format_message(question):
    answer_type = question["answer_type"]
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == "exact_match" else SYSTEM_MC
    question_text = question["question"]

    text_content = dict(type="text", text=question_text)
    if question["image"]:  # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question["image"]))
        content = [text_content, image_content]
    else:
        content = [text_content]

    # system_role = "user" if "o1" in args.model else "system" # o1 no sys prompt
    # messages = [
    #     {"role": system_role, "content": system_prompt},
    #     {"role": "user", "content": content}
    # ]
    content = system_prompt + "\n" + question_text
    messages = [{"role": "user", "content": content}]
    return messages


########################################################################


class HLESubsetBenchmark(BaseBenchmark):
    """
    Humanity's Last Exam benchmark for evaluating the "frontier of human knowledge".
    Link: https://huggingface.co/datasets/cais/hle
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
        Initialize HLE benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = max_tokens
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
                    "problem_id": str(example["id"]) if "id" in example else str(idx),
                    "expected_answer": str(example["answer"]),
                    "reference_rationale": str(example["rationale"]) if "rationale" in example else "",
                }

                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for HLE...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)
        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        examples_list = []

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [get_multiple_choice_answer(o) for o in outputs]
            examples_list.append(example)

        return {"examples": examples_list}

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

    def evaluate_responses_judge(
        self, results: Dict[str, Any], judge: str = "gpt-4o-mini-2024-07-18"
    ) -> Dict[str, float]:
        """
        Evaluate the generated solution completions using LM-Judge, as in original HLE.
        """

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        self.logger.info(f"Evaluating {num_questions} examples...")

        dataset = self.load_questions()
        if self.debug:
            dataset = dataset.select(range(2))
            self.logger.info(f"Debug mode: using 2 examples")

        dataset = dataset.to_dict()
        questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

        for example in examples:
            example["judge_responses"] = []

        for i in range(self.n_repeat):
            predictions = {example["id"]: {"response": example["model_outputs"][i]} for example in examples}

            eval_results = asyncio.run(judge_all_responses(questions, predictions, num_workers=2, judge=judge))

            for i, (unique_id, predictions) in enumerate(eval_results):
                if unique_id is not None:
                    examples[i]["judge_responses"].append(predictions["judge_response"])

        results.update(
            {
                "num_total": num_questions,
                "num_repeat": self.n_repeat,
            }
        )

        return results

    def load_questions(self) -> Dataset:
        """
        Load HLE questions from source.
        Keep only the multiplechoice and no images.
        """
        self.logger.info("Loading HLE questions from source, filtering for multiplechoice and no images...")
        dataset = load_dataset("cais/hle", split="test", cache_dir=HF_HUB_CACHE)
        dataset = dataset.filter(lambda x: x["answer_type"] == "multipleChoice")
        dataset = dataset.filter(lambda x: x["image"] == "")
        self.logger.info(f"{len(dataset)} examples remaining after filtering for multiplechoice and no images.")
        return dataset
