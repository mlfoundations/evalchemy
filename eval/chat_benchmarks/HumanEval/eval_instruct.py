from typing import Dict, List, Any, Optional
import json
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
import logging

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from human_eval.evaluation import evaluate_functional_correctness
from utils.utils import extract_generation_code, language_settings
from eval.task import BaseBenchmark


class HumanEvalBenchmark(BaseBenchmark):
    """
    HumanEval benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        languages: List[str] = ["python", "sh"],
        data_dir: str = "eval/chat_benchmarks/HumanEval/data",
        max_tokens: int = 1024,
        num_workers: int = 8,
        timeout: float = 3.0,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize HumanEval benchmark.

        Args:
            languages: List of programming languages to evaluate
            data_dir: Directory containing HumanEval datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.languages = languages
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.timeout = timeout
        self.debug = debug

    def build_deepseekcoder_instruction(self, language: str, question: str) -> str:
        """Build instruction prompt for the model."""
        return """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip().format(
            language.lower(), question.strip()
        )

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

        for lang in self.languages:
            try:
                problem_file = os.path.join(self.data_dir, f"humaneval-{lang}.jsonl")
                if not os.path.exists(problem_file):
                    self.logger.warning(f"Dataset file not found: {problem_file}")
                    continue

                examples = [json.loads(x) for x in open(problem_file) if x.strip()]
                self.logger.info(f"Loaded {len(examples)} examples for {lang}")

                if self.debug:
                    examples = examples[:2]
                    self.logger.info("Debug mode: using first 2 examples only")

                all_instances = []
                for idx, example in enumerate(examples):
                    prompt = self.build_deepseekcoder_instruction(
                        language_settings[lang]["full_name"], example["prompt"]
                    )
                    inputs = model.apply_chat_template([{"role": "user", "content": prompt}])

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
                self.logger.info("Generating responses for Human Eval...")
                outputs = self.compute(model, all_instances)

                is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
                if not is_main_process:
                    continue

                generated_examples = []
                for example, output in zip(examples, outputs):
                    example_with_output = example.copy()
                    example_with_output["output"] = output
                    processed_example = extract_generation_code(example_with_output, lang_code=lang)
                    generated_examples.append(processed_example)

                results[lang] = generated_examples
                temp_file_path = os.path.join(temp_dir, f"generated_{lang}.jsonl")
                with open(temp_file_path, "w", encoding="utf-8") as fw:
                    for ex in generated_examples:
                        fw.write(json.dumps(ex) + "\n")

                self.logger.info(f"Generated and saved {len(generated_examples)} examples for {lang}")

            except Exception as e:
                self.logger.error(f"Error processing language {lang}: {str(e)}")
                continue

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

        for lang in self.languages:
            try:
                problem_file = os.path.join(self.data_dir, f"humaneval-{lang}.jsonl")
                temp_file_path = os.path.join(temp_dir, f"generated_{lang}.jsonl")

                if not os.path.exists(temp_file_path):
                    self.logger.warning(f"Generated file not found: {temp_file_path}")
                    continue

                result = evaluate_functional_correctness(
                    input_file=temp_file_path,
                    tmp_dir=temp_dir,
                    n_workers=self.num_workers,
                    timeout=self.timeout,
                    problem_file=problem_file,
                    language=lang,
                )

                for metric, value in result.items():
                    evaluation_results[f"{lang}_{metric}"] = value

                self.logger.info(f"Completed evaluation for {lang}")

            except Exception as e:
                self.logger.error(f"Error evaluating {lang}: {str(e)}")
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
        self.logger.info(f"Running HumanEval benchmark for languages: {self.languages}")
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
