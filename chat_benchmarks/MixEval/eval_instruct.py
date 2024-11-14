from typing import Dict, List, Any, Optional
import logging
import torch
import os
import json
import time
import warnings
from argparse import Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from lm_eval.api.model import LM
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

        for split in splits:
            results = self._eval_split(model, split)
            if model.world_size > 1:
                # Gather results from all GPUs
                all_results = [None for _ in range(model.world_size)]
                dist.all_gather_object(all_results, results)

                if model.rank == 0:
                    # Combine results on main process
                    combined_results = []
                    for gpu_results in all_results:
                        combined_results.extend(gpu_results)
                    out_dict[split] = combined_results
                    response_file = self._get_response_file()
                    with open(response_file, "w") as f:
                        for result in combined_results:
                            f.write(json.dumps(result) + "\n")
            else:
                out_dict[split] = results

        if model.world_size > 1:
            dist.destroy_process_group()

        # Only return results on rank 0
        if model.world_size > 1 and model.rank != 0:
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
        if model.world_size > 1:
            # Add GPU rank to filename to avoid conflicts
            response_file = response_file.replace(".jsonl", f"_rank{model.rank}.jsonl")

        resume_info = self._check_resume(response_file)

        chat_model = self._get_model(model)
        eval_dataset = get_eval_dataset(self.args)

        if model.world_size > 1:
            sampler = DistributedSampler(eval_dataset, num_replicas=model.world_size, rank=model.rank)
            shuffle = False
        else:
            sampler = None
            shuffle = False

        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=32,
            collate_fn=lambda x: x,
            sampler=sampler,
        )

        time_elapsed = resume_info["time_elapsed"]
        start_time = time.time()

        with torch.no_grad():
            for b_id, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split}", unit="batch")):
                if resume_info["resume"] and resume_info["status"]["batch_id"] >= b_id:
                    continue

                chat_model.get_responses(batch, response_file)

                time_elapsed += time.time() - start_time
                start_time = time.time()

                self._update_status(b_id, time_elapsed, "in progress")

        self._update_status(b_id, time_elapsed, "complete")
        self.logger.info(
            f"Finished evaluating {self.args.model_name}'s {split} split. Used {round(time_elapsed / 60, 2)} minutes."
        )

        return self._load_results(response_file)

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
        if model.world_size > 1 and model.rank != 0:
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
