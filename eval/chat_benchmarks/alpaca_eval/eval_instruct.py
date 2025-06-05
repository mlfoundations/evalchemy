from typing import Dict, List, Any, Optional
import logging
import torch
import datasets
from tqdm import tqdm
import pandas as pd

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from alpaca_eval.main import evaluate as alpaca_eval_evaluate
from alpaca_eval.constants import DEFAULT_ANNOTATOR_CONFIG
from eval.task import BaseBenchmark


class AlpacaBenchmark(BaseBenchmark):
    """
    Alpaca benchmark for evaluating language model responses on instruction following.
    """

    REQUIRES_OPENAI_ANNOTATOR = True

    ANNOTATOR_CONFIG_MAP = {
        "gpt-4o-mini-2024-07-18": "weighted_alpaca_eval_gpt-4o-mini-2024-07-18",
        "gpt-4-1106-preview": "weighted_alpaca_eval_gpt4_turbo",
        "gpt-4": "alpaca_eval_gpt4_0613",
        "auto": DEFAULT_ANNOTATOR_CONFIG,
    }

    def __init__(
        self,
        dataset_name: str = "tatsu-lab/alpaca_eval",
        subset: str = "alpaca_eval",
        split: str = "eval",
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.5,
        do_sample: bool = True,
        debug: bool = False,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize Alpaca benchmark.

        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset name
            split: Dataset split to use
            max_tokens: Maximum number of tokens for generation
            temperature: Sampling temperature
            do_sample: Whether to use sampling for generation
            debug: debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.max_tokens = max_tokens if max_tokens is not None else 1024
        self.temperature = temperature
        self.do_sample = do_sample
        self.debug = debug
        self.annotator_conf = self.ANNOTATOR_CONFIG_MAP[annotator_model]

    def load_dataset(self) -> datasets.Dataset:
        """Load the evaluation dataset."""
        try:
            dataset = datasets.load_dataset(self.dataset_name, self.subset, trust_remote_code=True)[self.split]

            if self.debug:
                dataset = dataset.select(range(2))
                self.logger.info(f"Debug mode: using 2 examples")

            self.logger.info(f"Loaded {len(dataset)} examples for evaluation")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate completions for instructions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing model outputs and identifier
        """
        try:
            eval_set = self.load_dataset()

            all_instances = []
            for idx, example in enumerate(eval_set):
                try:
                    instruction = example["instruction"]
                    formatted_instruction = self._prepare_messages([{"role": "user", "content": instruction}], model)

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                formatted_instruction,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "do_sample": self.do_sample,
                                    "temperature": self.temperature,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    continue

            with torch.no_grad():
                self.logger.info("Generating responses for Alpaca Eval...")
                outputs = self.compute(model, all_instances)

            if model.rank != 0:
                return None

            model_outputs = []
            for idx, (example, output) in enumerate(zip(eval_set, outputs)):
                try:
                    instance = {
                        "instruction": example["instruction"],
                        "dataset": example["dataset"],
                        "datasplit": self.split,
                        "generator": model.model_identifier,
                        "output": output,
                    }
                    model_outputs.append(instance)
                except Exception as e:
                    self.logger.error(f"Error processing output {idx}: {str(e)}")
                    continue

            self.logger.info(f"Generated {len(model_outputs)} responses")

            return {"model_outputs": model_outputs, "model_identifier": model.model_identifier}

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using Alpaca evaluation metrics.

        Args:
            results: Dictionary containing model outputs and identifier

        Returns:
            Dictionary containing evaluation metrics
        """
        if results is None:
            return None

        model_outputs = results["model_outputs"]
        model_identifier = results["model_identifier"]

        if not model_outputs:
            raise ValueError("No model outputs to evaluate")

        self.logger.info("Running Alpaca evaluation...")
        leaderboard = alpaca_eval_evaluate(
            model_outputs=model_outputs,
            is_return_instead_of_print=True,
            is_overwrite_leaderboard=True,
            annotators_config=self.annotator_conf,
        )

        metrics = leaderboard[0].loc[model_identifier].to_dict()

        metrics.update(
            {
                "num_examples": len(model_outputs),
                "completion_rate": len(model_outputs) / len(results.get("total_examples", model_outputs)),
            }
        )

        self.logger.info("Evaluation complete")
        return metrics

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete Alpaca benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting Alpaca benchmark evaluation")
        try:
            generation_results = self.generate_responses(model)

            # If not rank 0, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)
            evaluation_results.update(
                {"benchmark_version": "alpaca_eval", "temperature": self.temperature, "max_tokens": self.max_tokens}
            )
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
