from typing import Dict, List, Any, Generator, Optional
import json
import logging
import tempfile
import os

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from eval.task import BaseBenchmark
from .evaluation import evaluate_accuracy


class IFEvalBenchmark(BaseBenchmark):
    def __init__(
        self,
        data_dir: str = "eval/chat_benchmarks/IFEval/data",
        max_tokens: int = 512,
        num_examples: int = 3,
        start_idx: int = 10,
        end_idx: int = 510,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Instruction Following Benchmark

        Args:
        data_dir: Directory containing MBPP datasets
        max_tokens: Maximum number of tokens for generation
        num_examples: Number of examples to show in few-shot prompt
        start_idx: Start index for evaluation examples
        end_idx: End index for evaluation examples
        debug_size: If set, only evaluate this many examples
        logger: Optional logger instance
        """
        super().__init__(logger)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_examples = num_examples
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.debug = debug

    def read_test_examples(self, data_path: str) -> Generator[Dict[str, str], None, None]:
        """
        Read and get prompts from data file.

        Args:
            data_path: Path to the data file

        Yields:
            Dictionary containing key, prompt and other properties required for evaluation
        """
        try:
            with open(data_path, "r") as f:
                examples = [json.loads(x) for x in f]

            self.logger.info(f"Loaded {len(examples)} examples from {data_path}")

            eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
            if self.debug:
                eval_range = list(eval_range)[:2]
                self.logger.info(f"Debug mode: using 2 examples")

            for i in eval_range:
                ex = examples[i]

                yield ex

        except Exception as e:
            self.logger.error(f"Error reading examples: {str(e)}")
            raise

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses to prompts using provided model

        Args:
            model: Language model instance

        Returns:
            Dictionary containing generated responses and temporary directory
        """
        try:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

            problem_file = os.path.join(self.data_dir, "input_data.jsonl")
            examples = list(self.read_test_examples(problem_file))
            self.logger.info(f"Process {len(examples)} examples")

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

            self.logger.info("Generating responses...")
            outputs = self.compute(model, all_instances)

            is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
            if not is_main_process:
                return None

            generated_examples = []
            for example, output in zip(examples, outputs):
                try:
                    example_with_output = example.copy()
                    example_with_output["response"] = output
                    generated_examples.append(example_with_output)

                except Exception as e:
                    self.logger.error(f"Error processing output for {example['key']}: {str(e)}")
                    continue

            output_path = os.path.join(temp_dir, "ifeval.jsonl")

            with open(output_path, "w", encoding="utf-8") as fw:
                for ex in generated_examples:
                    fw.write(json.dumps(ex) + "\n")

            return {
                "temp_dir_obj": temp_dir_obj,
                "num_examples": len(generated_examples),
                "total_examples": len(examples),
            }

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate generated responses for specific instructions

        Args:
            results: Dictionary containing generation responses

        Returns:
            Dictionary containing evaluation metrics -- i.e. strict and loose
        """
        try:
            temp_dir_obj = results["temp_dir_obj"]
            temp_dir = temp_dir_obj.name

            input_file = os.path.join(self.data_dir, "input_data.jsonl")
            response_file = os.path.join(temp_dir, "ifeval.jsonl")
            result = evaluate_accuracy(response_file)

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
        Run the complete Instruction Following Evaluation pipeline

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Starting IF benchmark evaluation")

        try:
            generation_results = self.generate_responses(model)
            evaluation_results = self.evaluate_responses(generaiton_results)

            evaluation_results.update(
                {"benchmark_version": "ifeval", "max_tokens": self.max_tokens, "num_shot": self.num_examples}
            )

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
