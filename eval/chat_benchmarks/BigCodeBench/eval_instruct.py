import json
import logging
import os
import tempfile
import traceback
from typing import Any, Dict, Optional, Union

from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

from .evaluation import evaluate_functional_correctness
from .sanitize import code_extract

try:
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
    )
except ImportError:
    print("Error importing vllm.distributed")

import gc

# suppress warnings
import warnings

import torch

warnings.filterwarnings("ignore")


BIGCODEBENCH_PATH = os.environ.get("BIGCODEBENCH_PATH", "eval/chat_benchmarks/BigCodeBench/data")
BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_VERSION = "v0.1.3"


def cleanup_resources():
    # Explicitly destroy distributed resources
    try:
        destroy_model_parallel()
    except Exception as e:
        print(f"Error in destroy_model_parallel: {e}")

    try:
        destroy_distributed_environment()
    except Exception as e:
        print(f"Error in destroy_distributed_environment: {e}")

    try:
        cleanup_dist_env_and_memory()
    except Exception as e:
        print(f"Error in cleanup_dist_env_and_memory: {e}")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()


class BigCodeBenchBenchmark(BaseBenchmark):
    """
    BigCodeBench benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        language: str = "python",
        data_dir: str = BIGCODEBENCH_PATH,
        max_tokens: int = 1280,
        num_workers: int = 32,
        timeout: float = 120,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        instruction_prefix: str = None,
        response_prefix: str = None,
        safe_mode: bool = False,
        check_ground_truth: bool = False,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize BigCodeBench benchmark.

        Args:
            languages: List of programming languages to evaluate
            data_dir: Directory containing BigCodeBench datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.language = language
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.timeout = timeout
        self.debug = debug
        self.prompt_types = ["instruct", "complete", "instruct-hard", "complete-hard"]
        self.safe_mode = safe_mode
        self.check_ground_truth = check_ground_truth
        if instruction_prefix is None:
            self.instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate code completions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        results = {}
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        for prompt_type in self.prompt_types:
            all_instances = []
            try:
                problem_file = os.path.join(self.data_dir, f"BigCodeBench-{prompt_type}.json")
                if not os.path.exists(problem_file):
                    self.logger.warning(f"Dataset file not found: {problem_file}")
                    continue

                examples = []

                with open(problem_file, "r") as fr:
                    examples = json.load(fr)

                if self.debug:
                    examples = examples[:2]
                    self.logger.info(f"Debug mode enabled. Using only {len(examples)} examples.")

                self.logger.info(f"Loaded {len(examples)} examples for BigCodeBench-{prompt_type}")

                formatted_inputs = []
                for example in examples:
                    prompt = example["prompt"].strip()
                    if prompt_type.startswith("complete"):
                        task_prompt = f"""\
                            {self.instruction_prefix}
                            ```
                            {prompt.strip()}
                            ```
                            """
                    else:
                        task_prompt = f"""\
                            {self.instruction_prefix}
                            {prompt.strip()}
                            """
                    inputs = task_prompt = self._prepare_messages([{"role": "user", "content": task_prompt}], model)
                    formatted_inputs.append(inputs)

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "temperature": 0,
                                },
                            ),
                            example["task_id"],
                        )
                    )

                self.logger.info(f"Generating responses for BigCodeBench-{prompt_type}...")
                if self.check_ground_truth:
                    # dummy outputs
                    self.logger.info(f"Using ground truth as output")
                    outputs = [example["canonical_solution"] for example in examples]
                else:
                    outputs = self.compute(model, all_instances)

                if model.rank != 0:
                    continue

                generated_examples = []
                for example, output in zip(examples, outputs):
                    example_with_output = example.copy()
                    example_with_output["output"] = output
                    example_with_output["prompt"] = example_with_output.pop("prompt")

                    example_with_output["generation"] = code_extract(example_with_output["output"])
                    if self.check_ground_truth:
                        example_with_output["generation"] = (
                            example["code_prompt"] + "\n" + example["canonical_solution"]
                        )
                    generated_examples.append(example_with_output)

                results[prompt_type] = generated_examples
                temp_file_path = os.path.join(temp_dir, f"generated_{prompt_type}.jsonl")

                with open(temp_file_path, "w", encoding="utf-8") as fw:
                    for ex in generated_examples:
                        fw.write(json.dumps(ex) + "\n")

                self.logger.info(f"Generated and saved {len(generated_examples)} examples for {prompt_type}")

            except Exception as e:
                self.logger.error(f"Error processing: {str(e)}")
                traceback.print_exc()

                continue

        try:
            cleanup_resources()
        except Exception as e:
            self.logger.warning(f"Error cleaning up resources: {str(e)}")
        results["temp_dir_obj"] = temp_dir_obj
        # results["temp_dir_obj"] = temp_dir
        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated code completions.

        Args:
            results: Dictionary containing generation results

        Returns:
            Dictionary containing evaluation metrics
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        temp_dir_obj = results["temp_dir_obj"]
        temp_dir = temp_dir_obj.name
        # temp_dir = results["temp_dir_obj"]

        evaluation_results = {}

        for prompt_type in self.prompt_types:
            try:
                problem_file = os.path.join(self.data_dir, f"BigCodeBench-{prompt_type}.json")
                temp_file_path = os.path.join(temp_dir, f"generated_{prompt_type}.jsonl")

                if not os.path.exists(temp_file_path):
                    self.logger.warning(f"Generated file not found: {temp_file_path}")
                    continue

                result = evaluate_functional_correctness(
                    input_file=temp_file_path,
                    tmp_dir=temp_dir,
                    n_workers=self.num_workers,
                    timeout=self.timeout,
                    problem_file=problem_file,
                    language=self.language,
                    safe_mode=self.safe_mode,
                )

                for metric, value in result.items():
                    evaluation_results[f"{prompt_type}_{metric}"] = value

                self.logger.info(f"Completed evaluation for {prompt_type}")

            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"Error evaluating {prompt_type}: {str(e)}")
                continue

        temp_dir_obj.cleanup()
        return evaluation_results

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation results, or None for non-primary ranks
        """
        self.logger.info(f"Running BigCodeBench benchmark for languages: {self.languages}")
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
