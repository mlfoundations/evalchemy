import json
import logging
import numpy as np
import re
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv

from eval.task import BaseBenchmark


class AIWBenchmark(BaseBenchmark):
    """
    AIW Benchmark for evaluating the math reasoning of LLMs.
    """

    TARGET_IDS = {577, 580, 581, 582, 583, 584, 559, 560, 561, 562, 563, 564, 637, 638, 639, 640, 641, 642}

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/AIW/data/aiw_data.json",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        n_trials: int = 100,  # Run 100 trials
    ):
        """
        Initialize AIW benchmark.

        Args:
            data_file: File containing the AIW dataset
            debug: If set, only evaluate on 2 examples
            seed: Random seed
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_trials = n_trials

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses
        """
        examples = self.load_questions()

        # Filter examples based on the target IDs
        examples = [ex for ex in examples if ex["id"] in self.TARGET_IDS]

        if not examples:
            self.logger.warning("No matching examples found for the given IDs.")
            return None

        # Store results for each ID
        id_results = {ex["id"]: [] for ex in examples}

        for t_id in range(self.n_trials):
            all_instances = []
            seed = [s + t_id for s in self.seed]
            for example in examples:
                messages = [{"role": "user", "content": example["prompt"]}]
                try:
                    templated_messages = self._prepare_messages(messages, model)
                except Exception as e:
                    print(f"Error preparing messages: {e}")
                    templated_messages = messages

                all_instances.append(
                    Instance(
                        "generate_until",
                        example,
                        (
                            templated_messages,
                            {
                                "do_sample": True,
                                "max_new_tokens": self.max_new_tokens,
                                "temperature": 1.0,
                                "seed": seed,
                            },
                        ),
                        example["id"],
                    )
                )

            # Generate model responses
            outputs = self.compute(model, all_instances)

            if model.rank != 0:
                return None

            # Store per-ID results
            for example, output in zip(examples, outputs):
                example["model_output"] = output
                example["model_answer"] = self.extract_answer(output)

        return {"examples": examples}

    def load_questions(self) -> List[Dict[str, str]]:
        """Load AIW questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = json.load(f)
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution."""
        try:
            return re.findall(r"answer:.*?(\d+)", output.lower())[-1]
        except:
            try:
                return re.findall(r"answer is.*?(\d+)", output.lower())[-1]
            except:
                try:
                    return re.findall(r"boxed{(\d+)}", output.lower())[-1]
                except:
                    return None

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions over multiple trials."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]

        # Filter only the target IDs
        examples = [ex for ex in examples if ex["id"] in self.TARGET_IDS]
        if not examples:
            self.logger.warning("No matching examples found for the given IDs.")
            return None

        # Store results per ID
        id_results = {ex["id"]: [] for ex in examples}

        for _ in range(self.n_trials):
            trial_results = []
            for example in examples:
                correct = is_equiv(str(example["right_answer"]), example["model_answer"])
                id_results[example["id"]].append(correct)
                trial_results.append(correct)

        # Compute per-ID statistics
        per_id_stats = {}
        for qid, id_result in id_results.items():
            results_array = np.array(id_result)  # Convert to numpy array
            mean_accuracy = np.mean(results_array)
            variance = np.var(results_array)
            per_id_stats[qid] = {"accuracy": mean_accuracy, "variance": variance}

        # Compute overall statistics
        all_accuracies = np.concatenate([id_results[qid] for qid in id_results])
        overall_mean_accuracy = np.mean(all_accuracies)
        overall_variance = np.var(all_accuracies)

        # Log results
        self.logger.info(f"Overall Accuracy: {overall_mean_accuracy:.4f}, Overall Variance: {overall_variance:.6f}")
        for qid, stats in per_id_stats.items():
            self.logger.info(f"ID {qid}: Accuracy = {stats['accuracy']:.4f}, Variance = {stats['variance']:.6f}")

        # Update results dictionary
        results.update(
            {
                "num_trials": self.n_trials,
                "num_examples": len(examples),
                "per_id_stats": per_id_stats,
                "overall_accuracy": overall_mean_accuracy,
                "overall_variance": overall_variance,
            }
        )

        return results
