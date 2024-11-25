from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import tempfile
import os
from pathlib import Path
import logging
import jsonlines
from datasets import load_dataset
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark

# Import WildBench utilities
from .src.unified_utils import save_outputs
from .src.eval import (
    compose_eval_item,
    batch_eval_generate,
    placeholder_generation,
)


@dataclass
class WildBenchConfig:
    """Configuration for WildBench evaluation."""

    # Dataset configuration
    data_name: str = "wild_bench"
    dataset_version: str = "v2"
    split: str = "test"
    start_idx: int = 0
    end_idx: int = -1

    # Model configuration
    max_tokens: int = 1024
    temperature: float = 0.0
    do_sample: bool = False
    engine: str = None
    model_name: str = None
    max_words_to_eval: int = 1000
    repetition_penalty: float = 1.0
    top_p: float = 1.0

    # Evaluation configuration
    model: str = None
    eval_template: str = "eval/chat_benchmarks/WildBench/evaluation/eval_template.score.v2.md"
    model: str = "gpt-4o-mini-2024-07-18"
    mode: str = "score"
    batch_mode: bool = True
    api_parallel: int = 32

    # Task weights
    task_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                "Creative Tasks": 0.5,
                "Planning & Reasoning": 1.25,
                "Math & Data Analysis": 1.0,
                "Information/Advice seeking": 0.75,
                "Coding & Debugging": 1.25,
            }


class WildBenchBenchmark(BaseBenchmark):
    """
    WildBench benchmark for evaluating diverse real-world tasks.
    """

    def __init__(
        self,
        config: Optional[WildBenchConfig] = None,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize WildBench benchmark.

        Args:
            config: WildBench configuration
            debug: If True, run in debug mode on 2 samples
            logger: Optional logger instance
        """
        super().__init__(logger)
        if annotator_model == "auto":
            annotator_model = "gpt-4-1106-preview"
        if config:
            self.logger.warning(f"Overwriting config.judge_model = {annotator_model} ")
            config.model = annotator_model
        self.config = config or WildBenchConfig(model=annotator_model)
        self.debug = debug

        # Task category mapping
        self.task_group_mapping = {
            "Information seeking": "Information/Advice seeking",
            "Creative Writing": "Creative Tasks",
            "Coding & Debugging": "Coding & Debugging",
            "Reasoning": "Planning & Reasoning",
            "Editing": "Creative Tasks",
            "Math": "Math & Data Analysis",
            "Planning": "Planning & Reasoning",
            "Brainstorming": "Creative Tasks",
            "Role playing": "Creative Tasks",
            "Advice seeking": "Information/Advice seeking",
            "Data Analysis": "Math & Data Analysis",
            "Others": "Creative Tasks",
        }

    def load_dataset(self) -> Tuple[List[str], List[Any], List[str], Dict[str, List[Any]]]:
        """Load and preprocess the evaluation dataset."""
        try:
            dataset = load_dataset("allenai/WildBench", self.config.dataset_version, split=self.config.split)

            if self.debug:
                dataset = dataset.select(range(min(2, len(dataset))))
                self.logger.info(f"Debug mode: using {len(dataset)} examples")

            # Initialize data structures
            chat_history = []
            id_strs = []
            extracted_chats = []
            metadata = {"session_id": [], "primary_tag": []}

            # Process each item
            for item in dataset:
                extracted_chats.append(item["conversation_input"])
                chat_history.append(item["conversation_input"])
                id_strs.append(item["session_id"])

                for key in metadata:
                    metadata[key].append(item[key])

            # Apply index limits
            if self.config.end_idx < 0:
                self.config.end_idx = len(id_strs)

            slice_range = slice(self.config.start_idx, self.config.end_idx)
            return (
                id_strs[slice_range],
                chat_history[slice_range],
                extracted_chats[slice_range],
                {k: v[slice_range] for k, v in metadata.items()},
            )

        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for WildBench tasks.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing file paths and temporary directory,
            or None for non-primary ranks
        """
        try:
            # Load data
            id_strs, chat_history, extracted_chats, metadata = self.load_dataset()

            # Prepare model inputs
            model_inputs = [model.apply_chat_template(chat) for chat in extracted_chats]

            # Create temporary directory
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name
            output_path = os.path.join(temp_dir, "output.json")

            # Generate responses
            all_instances = [
                Instance(
                    "generate_until",
                    None,
                    (
                        inputs,
                        {
                            "max_gen_toks": self.config.max_tokens,
                            "do_sample": self.config.do_sample,
                            "temperature": self.config.temperature,
                        },
                    ),
                    idx,
                )
                for idx, inputs in enumerate(model_inputs)
            ]

            self.logger.info("Generating responses for WildBench...")
            outputs = self.compute(model, all_instances)

            # Return None early for non-primary ranks
            is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
            if not is_main_process:
                return None

            outputs = [[output] for output in outputs]

            # Save outputs
            save_outputs(self.config, id_strs, outputs, chat_history, metadata, model_inputs, output_path)

            return {"filepath": output_path, "temp_dir_obj": temp_dir_obj}

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def process_evaluator_output(
        self, eval_result: List[Dict[str, Any]], task_mapping: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Process and calculate final scores from evaluator output."""
        lengths = []
        scores = []
        task_cat_results = {}

        for item in eval_result:
            if "score" not in item:
                continue

            score = float(item["score"])
            scores.append(score)

            # Process output length
            model_output = item["model_output"]
            if not model_output.endswith("... (truncated)"):
                output_len = len(model_output)
                if output_len > 0:
                    lengths.append(output_len)

            # Process task categories
            task_tags = task_mapping[item["session_id"]]
            for tag in task_tags:
                task_cat_results.setdefault(tag, []).append(score)

        # Calculate category scores
        task_cat_score = {tag: (sum(scores) / len(scores) - 5) * 2 for tag, scores in task_cat_results.items()}

        # Calculate weighted macro score
        task_macro_score = sum(task_cat_score[tag] * self.config.task_weights[tag] for tag in task_cat_score) / sum(
            self.config.task_weights.values()
        )

        return {
            "score": sum(scores) / len(scores),
            "adjusted_score": (sum(scores) / len(scores) - 5) * 2,
            "task_macro_score": task_macro_score,
            "adjusted_task_macro_score": task_macro_score,
            "task_categorized_scores": task_cat_score,
            "total_examples": len(eval_result),
            "avg_output_length": sum(lengths) / len(lengths) if lengths else 0,
        }

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated responses using GPT-4.

        Args:
            results: Dictionary containing generation results

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        try:
            temp_dir_obj = results["temp_dir_obj"]
            temp_dir = temp_dir_obj.name

            # Prepare evaluation files
            eval_file = os.path.join(temp_dir, "batch-submit.jsonl")

            # Load benchmark data
            bench_data = load_dataset("allenai/WildBench", self.config.dataset_version, split=self.config.split)

            # Load model outputs
            with open(results["filepath"], "r") as f:
                target_model_data = json.load(f)

            # Prepare evaluation items
            ref_model_data = [None] * len(target_model_data)
            histories = []
            last_queries = []
            checklists = []

            for b, t, r in zip(bench_data, target_model_data, ref_model_data):
                compose_eval_item(b, t, r, histories, last_queries, checklists)

            # Generate evaluation data
            eval_results = placeholder_generation(
                self.config, list(target_model_data), list(ref_model_data), histories, last_queries, checklists
            )

            # Generate batch evaluation
            json_lines = batch_eval_generate(eval_results, self.config)
            with open(eval_file, "w") as f:
                for line in json_lines:
                    f.write(json.dumps(line) + "\n")

            # Run GPT-4 evaluation
            client = OpenAI()
            self._process_evaluator_file(eval_file, client)

            # Process results
            submit_file = os.path.join(temp_dir, "results.jsonl")
            output_file = self._format_eval_file(submit_file, eval_file)

            # Create task mapping
            task_mapping = {}
            for item in bench_data:
                tags = [item["primary_tag"]] + item["secondary_tags"]
                task_mapping[item["id"]] = list(set(self.task_group_mapping[tag] for tag in tags))

            # Load and process evaluation results
            with open(output_file, "r") as f:
                eval_result = json.load(f)

            results = self.process_evaluator_output(eval_result, task_mapping)

            temp_dir_obj.cleanup()
            return results

        except Exception as e:
            self.logger.error(f"Error in evaluate_responses: {str(e)}")
            if "temp_dir_obj" in locals():
                temp_dir_obj.cleanup()
            raise

    def _process_evaluator_file(self, eval_file: str, client: OpenAI) -> None:
        """Process a single evaluation file with GPT-4."""
        try:
            with open(eval_file, "r") as file:
                lines = file.readlines()

            def process_line(line):
                payload = json.loads(line)
                payload["body"]["max_tokens"] = 4096
                response = client.chat.completions.create(**payload["body"])
                result = payload.copy()
                result["response"] = json.loads(response.json())
                return result

            results = []
            # Use ThreadPoolExecutor since API calls are I/O bound
            with ThreadPoolExecutor(max_workers=self.config.api_parallel) as executor:
                future_to_line = {executor.submit(process_line, line): line for line in lines}
                for future in as_completed(future_to_line):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing line: {str(e)}")

            output_file = eval_file.replace("batch-submit.jsonl", "results.jsonl")
            with open(output_file, "w") as file:
                for result in results:
                    file.write(json.dumps(result) + "\n")

        except Exception as e:
            self.logger.error(f"Error processing evaluator file: {str(e)}")
            raise

    def _format_eval_file(self, submit_file: str, eval_file: str) -> str:
        """Format raw evaluation results into structured output."""
        try:
            # Load submissions
            custom_id_to_submission = {}
            with jsonlines.open(submit_file, "r") as f:
                for line in f:
                    custom_id_to_submission[line["custom_id"]] = line

            # Process results
            results_json = []
            with jsonlines.open(submit_file, "r") as f:
                for item in f:
                    try:
                        custom_id = item["custom_id"]
                        submission = custom_id_to_submission[custom_id]
                        custom_id_splits = custom_id.split("||")
                        session_id = custom_id_splits[0]

                        eval_output = json.loads(item["response"]["choices"][0]["message"]["content"])

                        result_item = {
                            "session_id": session_id,
                            "parsed_result": eval_output,
                        }

                        # Extract model output
                        prompt = submission["body"]["messages"][0]["content"]
                        model_output = (
                            prompt.split("<|begin_of_response|>\n")[1].split("<|end_of_response|>\n")[0].strip()
                        )

                        # Add score information
                        model_test = custom_id_splits[1]
                        if "score" in eval_output:
                            result_item.update(
                                {"model_test": model_test, "score": eval_output["score"], "model_output": model_output}
                            )

                        results_json.append(result_item)

                    except Exception as e:
                        self.logger.warning(f"Error processing item: {str(e)}")
                        continue

            # Save formatted results
            output_file = eval_file.replace("batch_results.jsonl", "scores.json")
            with open(output_file, "w") as f:
                json.dump(results_json, f, indent=2)

            return output_file

        except Exception as e:
            self.logger.error(f"Error formatting eval file: {str(e)}")
            raise
