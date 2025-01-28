from collections import defaultdict
import json
import logging
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

import base64
import zlib
import pickle
import json
import copy
from .livecodebench_utils import lcb_run, map_to_example, has_test_type, post_process_code, translate_private_test_cases

from eval.task import BaseBenchmark
from eval.utils import SYSTEM_PROMPT
from datasets import load_dataset

import lm_eval.models
from lm_eval.models.vllm_causallms import VLLM


import re
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed


def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    return matches


class LiveCodeBenchBenchmark(BaseBenchmark):
    """
    LiveCodeBench Benchmark for evaluating the math reasoning of LLMs.

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize LiveCodeBench benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.debug = debug
        self.max_new_tokens = 32768  # set higher to avoid truncation for reasoning models

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
            examples = examples[:10]

        # Prepare instances for model
        all_instances = []
        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        else:
            model_name = model.model_args["model"]
        try:
            system_prompt = SYSTEM_PROMPT[model_name]
        except KeyError:
            system_prompt = "You are a helpful and harmless assistant. You should think step-by-step."
        for idx, example in enumerate(examples):
            if example["is_stdin"]:
                prompt_text = (
                    "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
                    + example["prompt"]
                )
            else:
                prompt_text = (
                    "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
                    + example["prompt"]
                )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}]
            templated_messages = model.apply_chat_template(messages)

            generation_args = {
                "do_sample": False,
                "max_gen_toks" if isinstance(model, VLLM) else "max_new_tokens": self.max_new_tokens,
                "temperature": 0.7,
            }

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (templated_messages, generation_args),
                    idx,
                )
            )

        # Generate model responses
        self.logger.info("Generating responses for LiveCodeBench...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        examples_list = []

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = has_code(output)
            examples_list.append(example)

        return {"examples": examples_list}

    @staticmethod
    def check_correctness(problem: Dict, completion: str, timeout: float, is_extracted: bool = False) -> Dict:
        """
        Evaluates the functional correctness of a completion by running the test
        suite provided in the problem.

        :param completion_id: an optional completion ID so we can match
            the results later even if execution finishes asynchronously.
        """
        result_list = lcb_run(problem, completion, timeout, is_extracted)
        details = [r[0] for r in result_list]
        all_passed = all(details)

        result = ""
        if result_list and all_passed:
            result = "passed"

        return result == "passed"

    def evaluate_single_example(self, example):
        """Helper function to evaluate a single example"""
        try:
            response_entry = {
                "content": example["model_answer"],
                "difficulty": example["difficulty"],
                "correctness": None,
                "reason": None,
            }

            code_filter_result = example["model_answer"]

            if not code_filter_result or len(code_filter_result) == 0:
                response_entry["correctness"] = False
                response_entry["reason"] = "Does not contain code component."
                return response_entry

            try:
                last_code = code_filter_result[-1]
                problem_to_check = copy.deepcopy(example)

                # Add debugging
                self.logger.debug(f"Evaluating {example['difficulty']} problem...")

                # Add timeout handling
                curr_res = self.check_correctness(
                    problem=problem_to_check,
                    completion=post_process_code(last_code),
                    timeout=6,
                    is_extracted=not problem_to_check["is_stdin"],
                )

                # Log the result
                self.logger.debug(f"Result for {example['difficulty']}: {curr_res}")

                response_entry["correctness"] = curr_res
                response_entry["reason"] = "" if curr_res else "Code is incorrect."

            except Exception as e:
                self.logger.error(f"Error evaluating {example['difficulty']} example: {str(e)}")
                response_entry["correctness"] = False
                response_entry["reason"] = f"Evaluation error: {str(e)}"

            return response_entry

        except Exception as outer_e:
            self.logger.error(f"Outer error in evaluate_single_example: {str(outer_e)}")
            return {
                "content": example.get("model_answer"),
                "difficulty": example.get("difficulty"),
                "correctness": False,
                "reason": f"Critical error: {str(outer_e)}",
            }

    def evaluate_responses(self, responses: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions in parallel using threads."""
        self.logger.info(f"Evaluating {len(responses['examples'])} examples...")

        # Use ThreadPoolExecutor with limited concurrency
        results = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_example = {}
            for i, example in enumerate(responses["examples"]):
                future = executor.submit(self.evaluate_single_example, example)
                future_to_example[future] = (i, example)

            # Collect results as they complete
            results = [None] * len(responses["examples"])
            for future in as_completed(future_to_example):
                idx, example = future_to_example[future]
                try:
                    result = future.result()
                    # Store both result and corresponding example
                    results[idx] = (result, example)
                    # self.logger.info(f"Example {idx} ({example['difficulty']}): {result['correctness']}")
                except Exception as e:
                    self.logger.error(f"Future error for example {idx}: {str(e)}")
                    results[idx] = (
                        {
                            "content": example["model_answer"],
                            "difficulty": example["difficulty"],
                            "correctness": False,
                            "reason": f"Future error: {str(e)}",
                        },
                        example,
                    )

        # Calculate metrics from results
        total_correct = sum(1 for result, _ in results if result["correctness"])
        total_finish = len(results)

        per_difficulty_correct = defaultdict(int)
        per_difficulty_total = defaultdict(int)

        for result, example in results:
            per_difficulty_correct[example["difficulty"]] += result["correctness"]
            per_difficulty_total[example["difficulty"]] += 1

        output = {
            "total_correct": total_correct,
            "total_finish": total_finish,
            "accuracy": total_correct / total_finish,
            "examples": [result for result, _ in results],  # Only include results in output
            "per_difficulty_correct": per_difficulty_correct,
            "per_difficulty_total": per_difficulty_total,
        }
        for difficulty in per_difficulty_correct.keys():
            output[f"accuracy_{difficulty}"] = per_difficulty_correct[difficulty] / per_difficulty_total[difficulty]

        self.logger.info(f"Total examples: {total_finish}")
        self.logger.info(f"Total correct: {total_correct}")
        self.logger.info(f"Overall accuracy: {total_correct / total_finish:.2%}")

        return output

    def load_questions(self) -> List[Dict[str, str]]:
        """Load LiveCodeBench questions from source."""
        # Load dataset in smaller chunks and combine
        all_examples = []
        chunk_size = 200  # Process 200 examples at a time

        for i in range(0, 511, chunk_size):  # Assuming total size is 511
            try:
                dataset = load_dataset(
                    "livecodebench/code_generation_lite",
                    version_tag="release_v2",
                    split=f"test[{i}:{i+chunk_size}]",
                    trust_remote_code=True,
                )

                # Process chunk
                dataset = dataset.map(
                    lambda example: {"private_test_cases": translate_private_test_cases(example["private_test_cases"])}
                )
                dataset = dataset.map(map_to_example, remove_columns=dataset.column_names)

                all_examples.extend(dataset)

            except ValueError:
                # We've reached the end of the dataset
                break

        return all_examples
