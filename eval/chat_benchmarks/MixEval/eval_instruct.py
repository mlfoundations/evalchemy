from typing import Dict, List, Any, Optional
import logging
import torch
import os
import json
import time
import warnings
from argparse import Namespace
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.models.dummy import DummyLM
from eval.task import BaseBenchmark

import mix_eval
from mix_eval.evaluate import parse_args
from mix_eval.utils.dataset import get_eval_dataset
from mix_eval.compute_metrics import compute_metrics_p
from mix_eval.utils.common_utils import cache_status, read_status, dict_equal
from mix_eval.models.lm_chat_model import LMChatModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MODEL_PARSER_API"] = os.getenv("OPENAI_API_KEY")

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)


class MixEvalBenchmark(BaseBenchmark):
    """
    MixEval benchmark for evaluating language model responses on various tasks.
    """

    def __init__(
        self,
        output_dir: str = "eval/chat_benchmarks/MixEval/results/",
        benchmark: str = "mixeval",
        version: str = "2024-06-01",
        batch_size: int = 8,
        max_gpu_memory: str = "60GB",
        data_path: str = "eval/chat_benchmarks/MixEval/mix_eval/data/",
        api_parallel_num: int = 32,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        verbose: bool = False,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MixEval benchmark.

        Args:
            output_dir: Directory to save results
            benchmark: Benchmark name
            version: Benchmark version
            batch_size: Batch size for evaluation
            max_gpu_memory: Maximum GPU memory to use
            data_path: Path to evaluation data
            api_parallel_num: Number of parallel API calls
            annotator_model: Model to use for multichoice judging and freeform judging
            verbose: Whether to print verbose output
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        os.makedirs(output_dir, exist_ok=True)
        if annotator_model == "auto":
            annotator_model = "gpt-3.5-turbo-0125"
        self.multichoice_judge = annotator_model
        self.freeform_judge = annotator_model
        self.args = self._get_args(
            {
                "output_dir": output_dir,
                "benchmark": benchmark,
                "version": version,
                "batch_size": batch_size,
                "max_gpu_memory": max_gpu_memory,
                "data_path": data_path,
                "api_parallel_num": api_parallel_num,
                "multichoice_judge": self.multichoice_judge,
                "freeform_judge": self.freeform_judge,
                "debug": debug,
                "verbose": verbose,
            }
        )

    def _get_args(self, params: Dict[str, Any]) -> Namespace:
        """
        Get default arguments and update with provided parameters.

        Args:
            params: Dictionary of parameter values

        Returns:
            Namespace object with all arguments
        """
        parser = parse_args(return_parser=True)
        default_args = {action.dest: action.default for action in parser._actions if action.dest != "help"}
        default_args.update(params)
        return Namespace(**default_args)

    def _get_model(self, model: LM) -> LMChatModel:
        """
        Get the appropriate model wrapper.

        Args:
            model: Language model instance

        Returns:
            LMChatModel instance
        """
        if isinstance(model, DummyLM):
            return mix_eval.api.registry.get_model(self.args.model_name)(self.args)
        else:
            return LMChatModel(self.args, model)

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses using distributed processing, but only for generation.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation results for each split
        """
        splits = ["close_freeform", "close_multichoice"]
        out_dict = {}

        self.logger.info("Generating responses for MixEval...")
        is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1

        for split in splits:
            self.args.split = split
            all_results = self._eval_split(model, split)
            if is_main_process:
                response_file = self._get_response_file()
                with open(response_file, "w") as f:
                    for result in all_results:
                        f.write(json.dumps(result) + "\n")
            out_dict[split] = all_results

        # Only return results on rank 0
        if not is_main_process:
            return None
        return out_dict

    def _eval_split(self, model: LM, split: str) -> List[Dict[str, Any]]:
        """
        Evaluate the model on a specific data split.

        Args:
            model: Language model instance
            split: Data split to evaluate on

        Returns:
            List of evaluation results for the specified split
        """
        self.args.split = split
        self.args.model_name = self._get_model_name(model)
        response_file = self._get_response_file()

        eval_dataset = get_eval_dataset(self.args)

        if self.args.debug:
            eval_dataset.raw_inputs = eval_dataset.raw_inputs[:2]
            self.logger.info(f"Debug mode: using 2 examples")

        all_prompts = [inp["formated_input"] for inp in eval_dataset.raw_inputs]
        all_instances = []

        for idx, instruction in enumerate(all_prompts):
            formatted_instruction = model.apply_chat_template([{"role": "user", "content": instruction}])

            all_instances.append(
                Instance(
                    "generate_until",
                    instruction,
                    (
                        formatted_instruction,
                        {
                            "temperature": 1.0,
                        },
                    ),
                    idx,
                )
            )
        all_responses = self.compute(model, all_instances)

        for idx in list(range(len(eval_dataset.raw_inputs))):
            eval_dataset.raw_inputs[idx]["response"] = all_responses[idx]

        is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
        if is_main_process:
            with open(response_file, "w") as f:
                for item in eval_dataset.raw_inputs:
                    json_line = json.dumps(item)
                    f.write(json_line + "\n")
        return eval_dataset.raw_inputs

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate responses only on rank 0

        Args:
            results: The model outputs to evaluate

        Returns:
            Dictionary containing evaluation metrics and samples
        """
        score_file = self._get_score_file()

        if not os.path.exists(score_file):
            compute_metrics_p(self.args)

        with open(score_file, "r") as f:
            metrics = json.load(f)

        judge_results = self._load_judge_results()

        return {
            self._get_judge_model_name(): {
                "metrics": metrics,
            },
            "samples": {
                "model_answers": results,
                "judge_answers": judge_results,
            },
        }

    def run_benchmark(self, model: LM) -> Dict[str, Any]:
        """
        Run benchmark with distributed generation but centralized evaluation

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics and samples
        """
        self.logger.info("Starting MixEval benchmark evaluation")
        generation_results = self.generate_responses(model)

        # Only evaluate on rank 0
        is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
        if not is_main_process:
            return None

        evaluation_results = self.evaluate_responses(generation_results)
        evaluation_results.update(
            {
                "benchmark_version": f"{self.args.benchmark}-{self.args.version}",
                "batch_size": self.args.batch_size,
                "max_gpu_memory": self.args.max_gpu_memory,
            }
        )
        return evaluation_results

    def _get_model_name(self, model: LM) -> str:
        if "model_identifier" in model.__dict__:
            return (
                model.model_identifier.split("pretrained=")[1]
                .split(",")[0]
                .split("__")[-1]
                .replace("Meta-", "")
                .replace("-", "_")
                .lower()
                .replace(".", "")
            )
        else:
            return model.model.__class__.__name__

    def _get_response_file(self) -> str:
        response_file = os.path.join(
            self.args.output_dir,
            self.args.model_name,
            self.args.benchmark,
            self.args.version,
            f"{self.args.model_name}_{self.args.split}.jsonl",
        )
        os.makedirs(os.path.dirname(response_file), exist_ok=True)
        return response_file

    def _check_resume(self, response_file: str) -> Dict[str, Any]:
        resume_info = {"resume": False, "time_elapsed": 0, "status": None}
        if os.path.exists(response_file):
            status = read_status(self.args)
            if self._args_match(status["args"]):
                if status["status"]["status"] == "complete":
                    self.logger.info(
                        f"The evaluation for {self.args.model_name}'s {self.args.split} split is already complete. Skipping."
                    )
                    resume_info["resume"] = True
                    resume_info["status"] = status["status"]
                else:
                    resume_info["resume"] = True
                    resume_info["time_elapsed"] = status["status"]["time_elapsed"]
                    resume_info["status"] = status["status"]
                    self.logger.info(f"Resuming from last run: \n{status}")
        return resume_info

    def _args_match(self, cached_args: Dict[str, Any]) -> bool:
        not_gen_args = [
            "freeform_judge",
            "multichoice_judge",
            "max_gpu_memory",
            "api_parallel_num",
            "max_tasks",
            "batch_size",
            "api_parallel_num",
        ]
        subdict_status = {k: v for k, v in cached_args.items() if k not in not_gen_args}
        subdict_args = {k: v for k, v in self.args.__dict__.items() if k not in not_gen_args}
        return dict_equal(subdict_status, subdict_args)

    def _update_status(self, batch_id: int, time_elapsed: float, status: str):
        cache_status(self.args, {"batch_id": batch_id, "time_elapsed": time_elapsed, "status": status})

    def _load_results(self, response_file: str) -> List[Dict[str, Any]]:
        with open(response_file, "r") as f:
            return [json.loads(line) for line in f]

    def _get_score_file(self) -> str:
        score_dir = os.path.join(self.args.output_dir, self.args.model_name, self.args.benchmark, self.args.version)
        judge_model = self._get_judge_model_name()
        return os.path.join(score_dir, f"score_{judge_model}.json")

    def _get_judge_model_name(self) -> str:
        return (
            self.args.multichoice_judge
            if self.args.multichoice_judge == self.args.freeform_judge
            else f"mc{self.args.multichoice_judge}_ff{self.args.freeform_judge}"
        )

    def _load_judge_results(self) -> Dict[str, List[Dict[str, Any]]]:
        score_dir = os.path.join(self.args.output_dir, self.args.model_name, self.args.benchmark, self.args.version)
        judge_results = {}
        for result_file in os.listdir(score_dir):
            if result_file.startswith("judge_results_ff"):
                with open(os.path.join(score_dir, result_file), "r") as f:
                    judge_results["close_freeform"] = [json.loads(line) for line in f]
            elif result_file.startswith("judge_results_mp"):
                with open(os.path.join(score_dir, result_file), "r") as f:
                    judge_results["close_multichoice"] = [json.loads(line) for line in f]
        return judge_results
