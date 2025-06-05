from typing import Dict, List, Any, Optional
import json
import os
import tempfile
from pathlib import Path
from tqdm import tqdm
import logging

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from multiple.evaluation import evaluate_functional_correctness
from .utils import extract_generation_code
from eval.task import BaseBenchmark
import traceback


LANUGUAGES = [
    # "adb",
    # "clj",
    "cpp",
    "cs",
    # "dart",
    # "dfy",
    # "dlang",
    # "elixir",
    # "fs",
    # "go",
    # "hs",
    "java",
    "js",
    # "julia",
    # "lean",
    "lua",
    # "luau",
    # "matlab",
    # "ocaml",
    "php",
    # "pl",
    # "python",
    "r",
    # "racket",
    # "ruby",
    "rs",
    # "scala",
    "sh",
    # "swift",
    "ts",
    # "v",
]

LANGUAGE_MAP = {
    "js": "javascript",
    "java": "java",
    "python": "python",
    "cpp": "cpp",
    "cs": "csharp",
    "go": "go",
    "hs": "haskell",
    "php": "php",
    "r": "r",
    "ruby": "ruby",
    "rs": "rust",
    "scala": "scala",
    "sh": "bash",
    "swift": "swift",
    "ts": "typescript",
    "adb": "ada",
    "clj": "clojure",
    "dart": "dart",
    "dfy": "fsharp",
    "dlang": "d",
    "elixir": "elixir",
    "fs": "fsharp",
    "julia": "julia",
    "lean": "lean",
    "lua": "lua",
    "luau": "lua",
    "matlab": "matlab",
    "ocaml": "ocaml",
    "pl": "perl",
    "racket": "racket",
    "v": "vlang",
}

DATA_DIR = "eval/chat_benchmarks/MultiPLE/data"


class MultipleBenchmark(BaseBenchmark):
    """
    Multipl-e benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        languages: List[str] = LANUGUAGES,
        data_dir: str = DATA_DIR,
        max_tokens: int = 1024,
        num_workers: int = 10,
        timeout: float = 15,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize multipl-e benchmark.

        Args:
            languages: List of programming languages to evaluate
            data_dir: Directory containing multipl-e datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.languages = languages
        self.data_dir = data_dir
        self.max_tokens = max_tokens or 1024
        self.num_workers = num_workers
        self.timeout = timeout
        self.debug = debug
        self.system_prompt = "You are a helpful programming assistant designed to complete code snippets."
        self.task_prompt = """Please generate code to complete the following problem:
        ```{lang}
        {prompt}
        ```
        """

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

        self.logger.info(f"Generating responses for MultiPL-E benchmark...")
        lang_str = ", ".join(self.languages)
        self.logger.info(f"Languages: {lang_str}")

        for lang in self.languages:
            try:
                problem_file = os.path.join(self.data_dir, f"multipl-e-{lang}.json")
                if not os.path.exists(problem_file):
                    self.logger.warning(f"Dataset file not found: {problem_file}")
                    continue

                with open(problem_file, "r", encoding="utf-8") as fr:
                    examples = json.load(fr)
                # examples = examples[:100]
                self.logger.info(f"Loaded {len(examples)} examples for {lang}")

                if self.debug:
                    examples = examples[:10]
                    self.logger.info("Debug mode: using first 2 examples only")

                all_instances = []
                for idx, example in enumerate(examples):
                    prompt = example["prompt"]
                    formatted_prompt = self.task_prompt.format(prompt=prompt, lang=LANGUAGE_MAP[lang])

                    inputs = model.apply_chat_template(
                        [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": formatted_prompt},
                        ]
                    )

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    # "do_sample": True,
                                    "temperature": 0.0,
                                },
                            ),
                            idx,
                        )
                    )
                self.logger.info(f"Generating responses for MulltiPL-E ({lang})...")
                outputs = self.compute(model, all_instances)

                if model.rank != 0:
                    continue

                generated_examples = []
                for example, output in zip(examples, outputs):
                    example_with_output = example.copy()
                    prompt = example["prompt"]
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
                problem_file = os.path.join(self.data_dir, f"multipl-e-{lang}.json")
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
                traceback.print_exc()
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
        self.logger.info(f"Running multipl-e benchmark for languages: {self.languages}")
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
