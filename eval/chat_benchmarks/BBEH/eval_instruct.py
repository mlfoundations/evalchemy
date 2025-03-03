import os
import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import hmean

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from collections import defaultdict
from eval.task import BaseBenchmark


import lm_eval.models
from lm_eval.models.vllm_causallms import VLLM

PROMPT = """{problem}

When you provide the final answer, please use the prefix "The answer is:" without any modification, and provide the answer directly, with no formatting and no markup. For instance: "The answer is: 42", or  "The answer is: yes", or "The answer is: (a)". For multi-choice questions, provide the letter corresponding to the correct answer. For instance: "The answer is: (a)"."""


class BBEHBenchmark(BaseBenchmark):
    """
    BBEH Benchmark for evaluating the math reasoning of LLMs.
    Link: https://github.com/google-deepmind/bbeh/tree/main
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/BBEH/data/",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize BBEH benchmark.

        Args:
            data_file: File containing the BBEH dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        # self.max_new_tokens = 32768  # set higher to avoid truncation for reasoning models
        self.max_new_tokens = 1024  # set higher to avoid truncation for reasoning models
        self.n_repeat = 1

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.load_questions()

        # Prepare instances for model
        all_instances = []
        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        else:
            model_name = model.model_args["model"]

        all_outputs = []
        for i in range(self.n_repeat):
            seed = [s + i for s in self.seed]
            all_instances = []
            for idx, example in enumerate(examples):
                messages = [
                    {"role": "user", "content": PROMPT.format(problem=example["input"])},
                ]

                templated_messages = model.apply_chat_template(messages)

                all_instances.append(
                    Instance(
                        "generate_until",
                        example,
                        (
                            templated_messages,
                            {
                                "do_sample": False,
                                "max_new_tokens": self.max_new_tokens,
                                "temperature": 0.7,
                                "seed": seed,
                            },
                        ),
                        idx,
                    )
                )

            # Generate model responses
            self.logger.info("Generating responses for BBEH...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(o) for o in outputs]

        return {"examples": examples}

        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            task_performance = defaultdict(int)
            task_performance_total = defaultdict(int)
            for example in examples:
                task = example["task"]
                task_performance[task] += (example["target"] == example["model_answers"][i])
                task_performance_total[task] += 1
            
            task_result = {}
            harmonic_mean_acc = []
            for task in task_performance:
                performance_task = 100 * task_performance[task] / task_performance_total[task]
                task_result[task] = performance_task
                harmonic_mean_acc.append(performance_task)
            
            ## Avoid division by zero
            harmonic_mean_acc  = [x + 1 if x == 0 else x for x in harmonic_mean_acc]
            harmonic_mean_acc = hmean(harmonic_mean_acc)
            task_result["harmonic_accuracy"] = harmonic_mean_acc
            task_result["repetition"] = i + 1
            all_results.append(task_result)

        # Calculate overall statistics
        harmonic_accuracy_avg = np.mean([result["harmonic_accuracy"] for result in all_results])
        harmonic_accuracy_std = np.std([result["harmonic_accuracy"] for result in all_results])
        harmonic_accuracy_std_err = np.std([result["harmonic_accuracy"] for result in all_results]) / np.sqrt(self.n_repeat)

        results.update(
            {
                "num_total": num_questions,
                "run_stats": all_results,
                "accuracy_avg": harmonic_accuracy_avg,
                "accuracy_std_err": harmonic_accuracy_std_err,
                "num_repeat": self.n_repeat,
            }
        )
        return results

    def load_questions(self) -> List[Dict[str, str]]:
        """Load BBEH questions from the data file."""

        subtasks = os.listdir(self.data_file)
        questions = []
        for subtask in subtasks:
            with open(os.path.join(self.data_file, subtask, "task.json"), "r") as f:
                subtask_data = json.load(f)['examples']
                subtask_data = [{"input": example["input"], "target": example["target"], "task": subtask} for example in subtask_data]
                questions = questions + subtask_data
        # import random
        # random.shuffle(questions)
        # questions = questions[:1000]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found in \boxed.
        """
        try:
            answer = output.split("The answer is:")[1].strip()
            if answer == "":
                answer = output.split("The answer is ")[1].strip()
            
            ## 3. -> 3
            if answer.endswith("."):
                answer = answer[:-1]

            return answer
        except:
            return ""
