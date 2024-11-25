from typing import Dict, List, Any, Optional
import tempfile
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from itertools import islice

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from archive_data.utils import load_data, construct_trainable_data, get_first_line_not_comment
from data.utils import construct_prompt
from evaluation.metrics import exact_match_score, edit_similarity_score, codebleu_score
from eval.task import BaseBenchmark


class RepoBenchmark(BaseBenchmark):
    """
    RepoBench benchmark implementation evaluating code completion capabilities
    across different programming languages and contexts.
    """

    def __init__(
        self,
        languages: List[str] = ["python", "java"],
        subsets: List[str] = ["cross_file_first", "cross_file_random", "in_file"],
        max_tokens: int = 2000,
        debug: bool = False,
        legacy_mode: bool = False,
    ):
        """
        Initialize RepoBench benchmark.

        Args:
            languages: List of programming languages to evaluate
            subsets: List of dataset subsets to use
            max_tokens: Maximum number of tokens for generation
            debug: If true, run on debug mode using 2 samples
            legacy_mode: Whether to use legacy (v0) evaluation
        """
        super().__init__()
        self.languages = languages
        self.subsets = subsets
        self.max_tokens = max_tokens
        self.debug = debug
        self.legacy_mode = legacy_mode

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate code completions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing temporary directory with generated responses,
            or None for non-primary ranks
        """
        if self.legacy_mode:
            return self._generate_responses_legacy(model)

        is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
        if is_main_process:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

        for lang in self.languages:
            datasets = load_dataset(f"tianyang/repobench_{lang}_v1.1", verification_mode="no_checks")

            for subset, dataset in datasets.items():
                if subset not in self.subsets:
                    continue

                if self.debug:
                    dataset = dataset.select(range(2))
                    self.logger.info(f"Debug mode: using 2 examples")

                all_instances = []
                # Split dataset across ranks for parallel construction
                # Get subset of dataset for this rank using the same slicing strategy as the compute function
                if hasattr(model, 'accelerator'):
                    chunk_size = len(dataset) // model.world_size
                    start = model.accelerator.process_index * chunk_size
                    end = start + chunk_size if model.accelerator.process_index < model.world_size - 1 else len(dataset)
                    rank_dataset = dataset.select(range(start, end))
                else:
                    rank_dataset = list(islice(dataset, model.rank, len(dataset), model.world_size))

                # Process examples for this rank's shard
                for idx, example in enumerate(rank_dataset):
                    prompt = construct_prompt(
                        example, tokenizer=model.tokenizer, max_token_nums=self.max_tokens, language=lang
                    )

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                prompt,
                                {"max_gen_toks": 128, "temperature": 0.2, "top_p": 0.95, "do_sample": True},
                            ),
                            idx,
                        )
                    )
                self.logger.info("Generating responses for RepoBench...")
                outputs = self.compute(model, all_instances, do_slice=False)

                # Only rank 0 should save the results
                is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
                if not is_main_process:
                    continue

                generated_examples = []
                for idx, (example, output) in enumerate(zip(dataset, outputs)):
                    generated_examples.append(
                        {
                            "idx": idx,
                            "gpt_completion": get_first_line_not_comment(output, language=lang),
                            "label": example["next_line"],
                        }
                    )

                output_path = f"{temp_dir}/repobench_{subset}_{lang}.jsonl"
                with open(output_path, "w", encoding="utf-8") as fw:
                    for ex in generated_examples:
                        fw.write(json.dumps(ex) + "\n")

        if is_main_process:
            return {"temp_dir_obj": temp_dir_obj}

    def _generate_responses_legacy(self, model: LM) -> Dict[str, Any]:
        """Legacy (v0) generation implementation."""
        prefix_token = "<fim_prefix>"
        suffix_token = "<fim_suffix><fim_middle>"

        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
        for lang in self.languages:
            for subset in self.subsets:
                dataset = load_data(split="test", task="completion", language=lang, length="2k", setting=subset)

                examples = construct_trainable_data(dataset, language=lang)

                all_instances = []
                for idx, example in enumerate(examples):
                    prompt = example["data"]

                    if "star" in model._model.config._name_or_path:
                        prompt = prefix_token + prompt + suffix_token

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                prompt,
                                {"max_new_tokens": 128, "temperature": 0.2, "top_p": 0.95, "do_sample": True},
                            ),
                            idx,
                        )
                    )

                outputs = self.compute(model, all_instances, do_slice=False)
                
                if not is_main_process:
                    continue

                generated_examples = []
                for idx, (example, output) in enumerate(zip(examples, outputs)):
                    generated_examples.append(
                        {
                            "idx": idx,
                            "gpt_completion": get_first_line_not_comment(output, language=lang),
                            "label": example["label"],
                        }
                    )

                output_path = f"{temp_dir}/repobench_{subset}_{lang}.jsonl"
                with open(output_path, "w", encoding="utf-8") as fw:
                    for ex in generated_examples:
                        fw.write(json.dumps(ex) + "\n")

        return {"temp_dir_obj": temp_dir_obj}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated code completions.

        Args:
            results: Dictionary containing temporary directory with generations

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        temp_dir_obj = results["temp_dir_obj"]
        temp_dir = temp_dir_obj.name

        evaluation_results = {}
        aggregated_stats = {"total_samples": 0, "total_em": 0, "total_es": 0}

        for lang in self.languages:
            for subset in self.subsets:
                filepath = os.path.join(temp_dir, f"repobench_{subset}_{lang}.jsonl")

                if not os.path.exists(filepath):
                    print(f"Missing file: {filepath}")
                    continue

                with open(filepath, "r") as f:
                    data = [json.loads(line) for line in f]

                if not data:
                    continue

                ground_truth = [d["label"] for d in data]
                generated = [d["gpt_completion"] for d in data]

                em_score = round(exact_match_score(ground_truth, generated) * 100, 2)
                es_score = round(edit_similarity_score(ground_truth, generated), 2)

                # Store individual scores
                evaluation_results[f"{lang}_{subset}_EM"] = em_score
                evaluation_results[f"{lang}_{subset}_ES"] = es_score

                # Update aggregated statistics
                aggregated_stats["total_samples"] += len(data)
                aggregated_stats["total_em"] += em_score * len(data)
                aggregated_stats["total_es"] += es_score * len(data)

                print(f"{lang}/{subset}: EM={em_score}, ES={es_score}, n={len(data)}")

        # Calculate weighted averages if we have data
        if aggregated_stats["total_samples"] > 0:
            for lang in self.languages:
                evaluation_results[f"{lang}_weighted_avg_EM"] = round(
                    aggregated_stats["total_em"] / aggregated_stats["total_samples"], 2
                )
                evaluation_results[f"{lang}_weighted_avg_ES"] = round(
                    aggregated_stats["total_es"] / aggregated_stats["total_samples"], 2
                )

        temp_dir_obj.cleanup()
        return evaluation_results

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
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
