import json
import logging
import os
import tempfile
import traceback
from typing import Any, Dict, Optional, Union

from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import re
import itertools

from eval.task import BaseBenchmark

from .evaluation import evaluate_generations

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

CruxEval_PATH = "eval/chat_benchmarks/CruxEval/data"


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


def make_cot_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""


def make_cot_input_prompt(s):
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""


def extract_answer(generation):
    if "[ANSWER]" in generation and "[/ANSWER]" in generation:
        start = generation.index("[ANSWER]") + len("[ANSWER]")
        end = generation.index("[/ANSWER]")
        return generation[start:end].strip()
    return generation


class CruxEvalBenchmark(BaseBenchmark):
    """
    CruxEval benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        data_dir: str = CruxEval_PATH,
        max_tokens: int = 2048,
        num_workers: int = 32,
        timeout: float = 120,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize CruxEval benchmark.

        Args:
            data_dir: Directory containing CruxEval datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.language = "python"
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.timeout = timeout
        self.debug = debug
        self.tasks = ["input", "output"]

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
        for task in self.tasks:
            all_instances = []
            try:
                problem_file = os.path.join(self.data_dir, f"cruxeval.jsonl")
                if not os.path.exists(problem_file):
                    self.logger.warning(f"Dataset file not found: {problem_file}")
                    continue

                examples = []

                with open(problem_file, "r", encoding="utf-8") as fr:
                    for line in fr:
                        examples.append(json.loads(line))

                if self.debug:
                    examples = examples[:10]
                    self.logger.info(f"Debug mode enabled. Using only {len(examples)} examples.")

                self.logger.info(f"Loaded {len(examples)} examples for CruxEval-{task}")

                formatted_inputs = []
                for example in examples:
                    code = example["code"]
                    inp = example["input"]
                    output = example["output"]
                    if task == "input":
                        task_prompt = make_cot_input_prompt((code, output))
                    else:
                        task_prompt = make_cot_output_prompt((code, inp))
                    inputs = self._prepare_messages([{"role": "user", "content": task_prompt}], model)
                    formatted_inputs.append(inputs)

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "do_sample": True,
                                    "temperature": 0.2,
                                },
                            ),
                            example["id"],
                        )
                    )

                self.logger.info(f"Generating responses for CruxEval-{task}...")

                outputs = self.compute(model, all_instances)
                if model.rank != 0:
                    continue

                generated_examples = []
                for example, output in zip(examples, outputs):
                    example_with_output = example.copy()

                    example_with_output["generation"] = extract_answer(output)
                    example_with_output["task_id"] = example_with_output.pop("id")

                    generated_examples.append(example_with_output)

                results[task] = generated_examples
                temp_file_path = os.path.join(temp_dir, f"generated_{task}.jsonl")

                with open(temp_file_path, "w", encoding="utf-8") as fw:
                    for ex in generated_examples:
                        fw.write(json.dumps(ex) + "\n")

                self.logger.info(f"Generated and saved {len(generated_examples)} examples for {task}")

            except Exception as e:
                self.logger.error(f"Error processing: {str(e)}")
                traceback.print_exc()

                continue

        try:
            cleanup_resources()
        except Exception as e:
            self.logger.warning(f"Error cleaning up resources: {str(e)}")
        results["temp_dir_obj"] = temp_dir_obj
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

        evaluation_results = {}

        for task in self.tasks:
            try:
                problem_file = os.path.join(self.data_dir, f"cruxeval.jsonl")
                temp_file_path = os.path.join(temp_dir, f"generated_{task}.jsonl")

                examples = []

                with open(problem_file, "r", encoding="utf-8") as fr:
                    for line in fr:
                        examples.append(json.loads(line))

                if self.debug:
                    examples = examples[:10]
                    self.logger.info(f"Debug mode enabled. Using only {len(examples)} examples.")

                if not os.path.exists(temp_file_path):
                    self.logger.warning(f"Generated file not found: {temp_file_path}")
                    continue

                result = evaluate_generations(
                    input_file=temp_file_path,
                    mode=task,
                    examples=examples,
                    tmp_dir=temp_dir,
                )

                for metric, value in result.items():
                    evaluation_results[f"{task}_{metric}"] = value

                self.logger.info(f"Completed evaluation for {task}")

            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"Error evaluating {task}: {str(e)}")
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
        self.logger.info(f"Running CruxEval benchmark for languages: {self.languages}")
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
