from typing import Dict, List, Any, Generator, Optional
import json
import os
import re
import tempfile
import logging
from tqdm import tqdm
from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from human_eval.evaluation import evaluate_functional_correctness
from eval.task import BaseBenchmark


class MBPPBenchmark(BaseBenchmark):
    """
    MBPP (Mostly Basic Python Programming) benchmark for evaluating
    Python code generation capabilities.
    """

    def __init__(
        self,
        data_dir: str = "eval/chat_benchmarks/MBPP/data",
        max_tokens: int = 512,
        num_examples: int = 3,
        start_idx: int = 10,
        end_idx: int = 510,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MBPP benchmark.

        Args:
            data_dir: Directory containing MBPP datasets
            max_tokens: Maximum number of tokens for generation
            num_examples: Number of examples to show in few-shot prompt
            start_idx: Start index for evaluation examples
            end_idx: End index for evaluation examples
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_examples = num_examples
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.debug = debug

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
                    i, self.format_test_example(ex["text"], ex["test_list"], ex["code"])
                )
                examples_str.append(example_prompt)

            eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
            if self.debug:
                eval_range = list(eval_range)[:2]
                self.logger.info(f"Debug mode: using 2 examples")

            for i in eval_range:
                ex = examples[i]
                prompt = self.format_test_example(ex["text"], ex["test_list"])

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

            problem_file = os.path.join(self.data_dir, "mbpp.jsonl")
            examples = list(self.read_test_examples(problem_file))
            self.logger.info(f"Processing {len(examples)} examples")

            all_instances = []
            for idx, example in enumerate(examples):
                try:
                    inputs = model.apply_chat_template([{"role": "user", "content": example["prompt"]}])

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_gen_toks": self.max_tokens,
                                    "do_sample": False,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    continue

            self.logger.info("Generating responses for MBPP...")
            outputs = self.compute(model, all_instances)

            # Return None early for non-primary ranks
            is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
            if not is_main_process:
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

            output_path = os.path.join(temp_dir, "mbpp.jsonl")
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

        try:
            temp_dir_obj = results["temp_dir_obj"]
            temp_dir = temp_dir_obj.name

            result = evaluate_functional_correctness(
                input_file=os.path.join(temp_dir, "mbpp.jsonl"),
                tmp_dir=temp_dir,
                problem_file=os.path.join(self.data_dir, "mbpp_test.jsonl"),
                language="python",
                is_mbpp=True,
            )

            result.update(
                {
                    "num_examples": results["num_examples"],
                    "completion_rate": results["num_examples"] / results["total_examples"],
                }
            )

            temp_dir_obj.cleanup()
            return result

        except Exception as e:
            self.logger.error(f"Error in evaluate_responses: {str(e)}")
            if temp_dir_obj:
                temp_dir_obj.cleanup()
            raise

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete MBPP benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting MBPP benchmark evaluation")
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)

            evaluation_results.update(
                {"benchmark_version": "mbpp", "max_tokens": self.max_tokens, "num_shot": self.num_examples}
            )

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
