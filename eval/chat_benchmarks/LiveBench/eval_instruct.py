from typing import Dict, Any, Optional, List
import logging
import subprocess
import os
from eval.chat_benchmarks.LiveBench.livebench.common import load_questions
from eval.chat_benchmarks.LiveBench.livebench.gen_ground_truth_judgment import play_a_match_gt
from eval.chat_benchmarks.LiveBench.livebench.model import get_conversation_template
from fastchat.utils import str_to_torch_dtype
import torch
import json
import shortuuid
import time
from tqdm import tqdm
from lm_eval.api.instance import Instance

from lm_eval.api.model import LM

import argparse
import json
import os
import random
import time
import glob

import shortuuid
import torch
from tqdm import tqdm

from eval.chat_benchmarks.LiveBench.livebench.common import (
    reorg_answer_file,
    get_categories_tasks,
    get_hf_dataset,
    get_tasks_from_hf_category,
    load_questions,
    load_questions_jsonl,
    LIVE_BENCH_DATA_SUPER_PATH,
)
from eval.chat_benchmarks.LiveBench.livebench.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from eval.chat_benchmarks.LiveBench.livebench.gen_api_answer import get_answer
from eval.chat_benchmarks.LiveBench.livebench.gen_ground_truth_judgment import gen_judgments
from eval.task import BaseBenchmark

class LiveBenchBenchmark(BaseBenchmark):
    """
    LiveBench benchmark for evaluating language model responses.
    """

    def __init__(
        self,
        dtype: str = "float32",
        max_new_token: int = 4096,
        dataset_name: str = "live_bench",
        question_source: str = "huggingface",
        temperature: float = 0.0,
        do_sample: bool = True,
        debug: bool = False,
        num_choices: int = 1,
        release_date: str = "2024-08-31",
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        remove_existing_file: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize LiveBench benchmark.

        Args:
            dtype: Data type for model inference
            dataset_name: Name of the dataset
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.dtype = dtype
        self.dataset_name = dataset_name
        self.question_source = question_source
        self.do_sample = do_sample
        self.debug = debug
        self.annotator_model = annotator_model
        self.release_date = release_date
        self.remove_existing_file = remove_existing_file
        self.num_workers = 32
        if self.debug:
            self.question_begin = 0
            self.question_end = 10
            self.max_tokens = 128
            self.release_date = "2024-06-24"
            self.num_workers = 0
        else:
            self.question_begin = None
            self.question_end = None
            self.max_tokens = max_new_token
        assert release_date in ["2024-07-26", "2024-06-24", "2024-08-31"]
        self.temperature = temperature
        self.num_choices = num_choices
        
        self.data_path = f"eval/chat_benchmarks/LiveBench/data"

    def get_question_list(self, model_name: str, release_set: set):
        questions_all = []
        answer_files  = []

        if self.question_source == "huggingface":
            categories, tasks = get_categories_tasks(self.dataset_name)

            for category_name, task_names in tasks.items():
                for task_name in task_names:
                    questions = load_questions(categories[category_name], release_set, task_name, self.question_begin, self.question_end)

                    task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                    answer_file = f"{self.data_path}/{task_full_name}/model_answer/{model_name}.jsonl"

                    questions_all.extend(
                        [
                            (q, answer_file)
                            for q in questions
                        ]
                    )

                    answer_files.append(answer_file)
        elif self.question_source == "jsonl":
            list_of_question_files = []
            original_question_file = f"{self.data_path}/{self.dataset_name}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(f"{self.data_path}/{self.dataset_name}/**/question.jsonl", recursive=True)

            for question_file in list_of_question_files:
                print(question_file)
                questions = load_questions_jsonl(question_file, release_set, self.question_begin, self.question_end)

                bench_name = os.path.dirname(question_file).replace(f"{self.data_path}/","")
                answer_file = f"{self.data_path}/{bench_name}/model_answer/{model_name}.jsonl"

                questions_all.extend(
                    [
                        (q, answer_file)
                        for q in questions
                    ]
                )

                if len(questions) > 0:
                    answer_files.append(answer_file)

        else:
            raise ValueError(f"Bad question source {self.question_source}.")

        questions_all = [
            q for q in questions_all if q[0]['livebench_removal_date'] == "" or q[0]['livebench_removal_date'] > self.release_date
        ]
        return questions_all

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

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate model answers using LiveBench.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing model outputs and identifier
        """
        self.logger.info("Generating responses for LiveBench...")
        # return [{"model_id": self._get_model_name(model)}]
        # Load questions
        model_name = self._get_model_name(model)
        questions = self.get_question_list(model_name, [self.release_date])
        questions = questions[self.question_begin:self.question_end]
        # Generate answers
        answers = []
        all_instances = []
        all_convs = [[] for _ in questions]
        all_choices = [{"index": i, "turns": []} for _ in questions for i in range(self.num_choices)]
        max_turns = max(len(q["turns"]) for q, _ in questions)
        for choice_num in range(self.num_choices):
            for turn_num in range(max_turns):
                for idx, (question, answer_file) in enumerate(tqdm(questions)):
                    if turn_num < len(question["turns"]):
                        qs = question["turns"][turn_num]
                        all_convs[idx].append({"role": "user", "content": qs})
                        
                        prompt = model.apply_chat_template(all_convs[idx])
                        
                        all_instances.append(
                            Instance(
                                "generate_until",
                                all_convs[idx],
                                (
                                    prompt,
                                    {
                                        "max_gen_toks": self.max_tokens,
                                        "do_sample": self.temperature >= 1e-4,
                                        "temperature": self.temperature,
                                    },
                                ),
                                idx,
                            )
                        )
                    
                    if all_instances:
                        outputs = self.compute(model, all_instances)

                        for idx, output in enumerate(outputs):
                            all_convs[idx].append({"role": "assistant", "content": output})
                            all_choices[choice_num]["turns"].append(output)
                      
                      
        if model.rank != 0:
            return all_choices
        
        results = []
        for idx, (question, answer_file) in enumerate(questions):
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "question": question,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_name,
                    "choices": all_choices,
                    "tstamp": time.time(),
                }
                results.append(ans_json)
                fout.write(json.dumps(ans_json) + "\n")

        return results

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using LiveBench evaluation metrics.

        Args:
            results: Dictionary containing model outputs and identifier

        Returns:
            Dictionary containing evaluation metrics
        """
        model_name = results[0]["model_id"]
        # questions = self.get_question_list(model_name, [self.release_date])
        if self.question_source == "huggingface":
            categories, tasks = get_categories_tasks(self.dataset_name)
            if self.debug:
                tasks = {"coding": ["coding_completion"]}
                categories = {"coding": categories["coding"]}

            for category_name, task_names in tasks.items():
                for task_name in task_names:
                    questions = load_questions(categories[category_name], self.release_date, task_name, self.question_begin, self.question_end)

                    questions = [
                        q for q in questions if q['livebench_removal_date'] == "" or q['livebench_removal_date'] > self.release_date
                    ]

                    task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                    output_file = f"{self.data_path}/{task_full_name}/model_judgment/ground_truth_judgment.jsonl"

                    answer_dir = f"{self.data_path}/{task_full_name}/model_answer/"

                    if len(questions) > 0:
                        gen_judgments(
                            parallel=self.num_workers,
                            questions=questions,
                            output_file=output_file,
                            answer_dir=answer_dir,
                            model_list=[model_name],
                            remove_existing_file=self.remove_existing_file,
                            bench_name=task_full_name,
                        )


        elif self.question_source == "jsonl":
            list_of_question_files = []
            original_question_file = f"{self.data_path}/{self.dataset_name}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(f"{self.data_path}/{self.dataset_name}/**/question.jsonl", recursive=True)

            for question_file in list_of_question_files:
                print(question_file)
                questions = load_questions_jsonl(question_file, self.release_date, self.question_begin, self.question_end)

                questions = [
                    q for q in questions if q['livebench_removal_date'] == "" or q['livebench_removal_date'] > self.release_date
                ]

                bench_name = os.path.dirname(question_file).replace(f"{self.data_path}/","")

                output_file = f"{self.data_path}/{bench_name}/model_judgment/ground_truth_judgment.jsonl"
                answer_dir = f"{self.data_path}/{bench_name}/model_answer/"
                if len(questions) > 0:
                    gen_judgments(
                        parallel=self.num_workers,
                        output_file=output_file,
                        answer_dir=answer_dir,
                        model_list=[model_name],
                        remove_existing_file=self.remove_existing_file,
                        bench_name=bench_name,
                    )

        else:
            raise ValueError(f"Bad question source {self.question_source}.")
        

    def run_benchmark(self) -> Dict[str, float]:
        """
        Run the complete LiveBench benchmark evaluation pipeline.

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Starting LiveBench benchmark evaluation")
        try:
            generation_results = self.generate_responses()

            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)
            evaluation_results.update(
                {"benchmark_version": "live_bench"}
            )
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
