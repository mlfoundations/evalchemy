from typing import Dict, List, Any, Optional, Generator
import json
import os
import re
import tempfile
from pathlib import Path
from tqdm import tqdm
import logging

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from mbpp_plus.evaluation import evaluate_functional_correctness
from .utils.utils import extract_generation_code, language_settings
from eval.task import BaseBenchmark


class MBPPPlusBenchmark(BaseBenchmark):
    """
    MBPPPlus benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        data_dir: str = "eval/chat_benchmarks/MBPPPlus/data",
        num_workers: int = 8,
        timeout: float = 3.0,
        debug: bool = False,
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MBPPPlus benchmark.

        Args:
            data_dir: Directory containing MBPPPlus datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.timeout = timeout
        self.debug = debug
        self.num_examples = 3
        self.start_idx = 0
        self.end_idx = 500

    def format_test_example(self, question: str, tests: List[str], code: Optional[str] = None) -> str:
        """Format a single test example."""
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(question.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    def read_test_examples(self, data_path: str) -> Generator[Dict[str, str], None, None]:
        """
        Read and format test examples from data file.

        Args:
            data_path: Path to the data file

        Yields:
            Dictionary containing task_id and formatted prompt
        """
        try:
            with open(data_path, "r") as f:
                examples = [json.loads(x) for x in f]
            self.logger.info(f"Loaded {len(examples)} examples from {data_path}")

            examples_str = []
            for i in range(1, self.num_examples + 1):
                ex = examples[i]
                example_prompt = "- Example {}:\n{}".format(
                    i, self.format_test_example(ex["prompt"], ex["test_list"], ex["code"])
                )
                examples_str.append(example_prompt)

            eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
            if self.debug:
                eval_range = list(eval_range)[:2]
                self.logger.info(f"Debug mode: using 2 examples")

            for i in eval_range:
                ex = examples[i]
                prompt = self.format_test_example(ex["prompt"], ex["test_list"])

                prompt_with_shots = """
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
""".strip().format(
                    "\n\n".join(examples_str), prompt
                )

                yield {"task_id": ex["task_id"], "prompt": prompt_with_shots}

        except Exception as e:
            self.logger.error(f"Error reading examples: {str(e)}")
            raise

    def extract_code(self, completion: str) -> str:
        """Extract code block from model completion."""
        try:
            code_block = re.findall(r"```python\n(.*?)```", completion, re.DOTALL | re.IGNORECASE)[0]
            return code_block
        except Exception as e:
            self.logger.warning(f"Failed to extract code block, using full completion.\nError: {str(e)}")
            return completion

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate code completions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        try:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

            problem_file = os.path.join(self.data_dir, "mbppplus.jsonl")
            examples = list(self.read_test_examples(problem_file))
            self.logger.info(f"Processing {len(examples)} examples")

            all_instances = []
            for idx, example in enumerate(examples):
                try:
                    inputs = self._prepare_messages([{"role": "user", "content": example["prompt"]}], model)

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "do_sample": False,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    continue

            self.logger.info("Generating responses for MBPPPlus...")
            outputs = self.compute(model, all_instances)

            # Return None early for non-primary ranks
            if model.rank != 0:
                return None

            generated_examples = []
            for example, output in zip(examples, outputs):
                try:
                    example_with_output = example.copy()
                    example_with_output["gpt_completion"] = output
                    example_with_output["generation"] = self.extract_code(output)
                    generated_examples.append(example_with_output)
                except Exception as e:
                    self.logger.error(f"Error processing output for {example['task_id']}: {str(e)}")
                    continue

            output_path = os.path.join(temp_dir, "generated_python.jsonl")
            with open(output_path, "w", encoding="utf-8") as fw:
                for ex in generated_examples:
                    fw.write(json.dumps(ex) + "\n")

            self.logger.info(f"Saved {len(generated_examples)} examples to {output_path}")

            return {
                "temp_dir_obj": temp_dir_obj,
                "num_examples": len(generated_examples),
                "total_examples": len(examples),
            }

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

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

        problem_file = os.path.join(self.data_dir, f"mbppplus.jsonl")
        temp_file_path = os.path.join(temp_dir, f"generated_python.jsonl")

        if not os.path.exists(temp_file_path):
            self.logger.warning(f"Generated file not found: {temp_file_path}")

        result = evaluate_functional_correctness(
            input_file=temp_file_path,
            tmp_dir=temp_dir,
            n_workers=self.num_workers,
            timeout=self.timeout,
            problem_file=problem_file,
            language="python",
        )

        for metric, value in result.items():
            evaluation_results[f"{metric}"] = value

        self.logger.info(f"Completed evaluation")

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
        self.logger.info(f"Running MBPPPlus benchmark")
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
