from typing import Dict, Any, Optional, List
import logging
import os
from eval.chat_benchmarks.LiveBench.livebench.common import load_questions
import json
import shortuuid
import time
from tqdm import tqdm
from lm_eval.api.instance import Instance

from lm_eval.api.model import LM
import lm_eval.models as lm_eval_models

import json
import os
import time
import glob

import shortuuid
from tqdm import tqdm

from eval.chat_benchmarks.LiveBench.livebench.common import (
    get_categories_tasks,
    load_questions,
    load_questions_jsonl,
    LIVE_BENCH_DATA_SUPER_PATH,
    reorg_answer_file,
)
from eval.chat_benchmarks.LiveBench.livebench.gen_ground_truth_judgment import gen_judgments
from eval.task import BaseBenchmark
from eval.chat_benchmarks.LiveBench.livebench.model import get_conversation_template


class LiveBenchBenchmark(BaseBenchmark):
    """
    LiveBench benchmark for evaluating language model responses.
    """

    def __init__(
        self,
        dataset_name: str = "live_bench",
        question_source: str = "huggingface",
        temperature: float = 0.0,
        do_sample: bool = True,
        debug: bool = False,
        num_choices: int = 1,
        release_date: str = "2024-08-31",
        remove_existing_file: bool = True,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 4096,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize LiveBench benchmark.

        Args:
            dataset_name: Name of the dataset
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.question_source = question_source
        self.do_sample = do_sample
        self.debug = debug
        self.release_date = release_date
        self.remove_existing_file = remove_existing_file
        self.num_workers = 1
        if self.debug:
            self.max_tokens = 128
            self.release_date = "2024-06-24"
            self.num_workers = 1
        else:
            self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_choices = num_choices
        self.all_release_dates = ["2024-07-26", "2024-06-24", "2024-08-31", "2024-11-25"]
        self.seed = seed
        self.data_path = f"eval/chat_benchmarks/LiveBench/data"

    def get_question_list(self, model_name: str, release_set: set):
        questions_all = []
        answer_files = []

        if self.question_source == "huggingface":
            categories, tasks = get_categories_tasks(self.dataset_name)

            for category_name, task_names in tasks.items():
                for task_name in task_names:
                    questions = load_questions(
                        categories[category_name],
                        release_set,
                        self.release_date,
                        task_name,
                    )
                    if self.debug:
                        questions = questions[:10]

                    task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                    answer_file = f"{self.data_path}/{task_full_name}/model_answer/{model_name}.jsonl"

                    questions_all.extend([(q, answer_file) for q in questions])

                    answer_files.append(answer_file)
        elif self.question_source == "jsonl":
            list_of_question_files = []
            original_question_file = f"{self.data_path}/{self.dataset_name}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(
                    f"{self.data_path}/{self.dataset_name}/**/question.jsonl", recursive=True
                )

            for question_file in list_of_question_files:
                questions = load_questions_jsonl(question_file, self.all_release_dates, self.release_date)
                if self.debug:
                    questions = questions[:10]

                bench_name = os.path.dirname(question_file).replace(f"{self.data_path}/", "")
                answer_file = f"{self.data_path}/{bench_name}/model_answer/{model_name}.jsonl"

                questions_all.extend([(q, answer_file) for q in questions])

                if len(questions) > 0:
                    answer_files.append(answer_file)

        else:
            raise ValueError(f"Bad question source {self.question_source}.")

        return questions_all

    def _get_model_name(self, model: LM) -> str:
        if "model_identifier" in model.__dict__:
            return (
                model.model_identifier.split("=")[1]
                .split(",")[0]
                .split("__")[-1]
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
        # Load questions
        model_name = self._get_model_name(model)
        questions = self.get_question_list(model_name, self.all_release_dates)

        # Generate answers
        all_choices = {}
        for idx, (_, answer_file) in enumerate(questions):
            if answer_file not in all_choices:
                all_choices[answer_file] = {}
            all_choices[answer_file][idx] = [{"index": i, "turns": []} for i in range(self.num_choices)]
        all_instances = []
        max_turns = max(len(q["turns"]) for q, _ in questions)
        if model.rank == 0:
            tqdm_gen = tqdm(range(self.num_choices * max_turns * len(questions)))
        for choice_num in range(self.num_choices):
            all_convs = [get_conversation_template(model_name) for _ in questions]
            for turn_num in range(max_turns):
                for idx, (question, answer_file) in enumerate(questions):
                    if turn_num < len(question["turns"]):
                        qs = question["turns"][turn_num]
                        all_convs[idx].append_message(all_convs[idx].roles[0], qs)
                        all_convs[idx].append_message(all_convs[idx].roles[1], None)

                        messages = [
                            {"role": "user", "content": all_convs[idx].get_prompt()},
                        ]
                        templated_messages = self._prepare_messages(messages, model)

                        all_instances.append(
                            Instance(
                                "generate_until",
                                all_convs[idx],
                                (
                                    templated_messages,
                                    {
                                        "max_new_tokens": self.max_tokens,
                                        "do_sample": self.temperature >= 1e-4,
                                        "temperature": self.temperature,
                                        "seed": self.seed,
                                    },
                                ),
                                idx,
                            )
                        )
                    if model.rank == 0:
                        tqdm_gen.update(1)
                        tqdm_gen.set_description(f"Generating {choice_num} {turn_num} {idx}")
                        tqdm_gen.refresh()

                if all_instances:
                    print(f"Computing... {len(questions)}")
                    outputs = self.compute(model, all_instances)

                    for idx, output in enumerate(outputs):
                        # Match gen_model_answer.py output cleaning
                        output = output.strip()

                        # Handle stop strings like in gen_model_answer.py
                        if all_convs[idx].stop_str and isinstance(all_convs[idx].stop_str, list):
                            stop_str_indices = sorted(
                                [
                                    output.find(stop_str)
                                    for stop_str in all_convs[idx].stop_str
                                    if output.find(stop_str) > 0
                                ]
                            )
                            if len(stop_str_indices) > 0:
                                output = output[: stop_str_indices[0]]
                        elif all_convs[idx].stop_str and output.find(all_convs[idx].stop_str) > 0:
                            output = output[: output.find(all_convs[idx].stop_str)]

                        # Handle special tokens like in gen_model_answer.py
                        if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "special_tokens_map"):
                            for special_token in model.tokenizer.special_tokens_map.values():
                                if isinstance(special_token, list):
                                    for special_tok in special_token:
                                        output = output.replace(special_tok, "")
                                else:
                                    output = output.replace(special_token, "")

                        # Handle xgen specific case like in gen_model_answer.py
                        if all_convs[idx].name == "xgen" and output.startswith("Assistant:"):
                            output = output.replace("Assistant:", "", 1).strip()

                        output = output.strip()
                        all_convs[idx].update_last_message(output)
            for idx, (_, answer_file) in enumerate(questions):
                choice_idx = choice_num  # Current choice index
                all_choices[answer_file][idx][choice_idx]["turns"] = [
                    c[1] for c in all_convs[idx].messages[2:] if c[0].lower() == "assistant"
                ]

        if model.rank != 0:
            return all_choices

        results = []
        for idx, (question, answer_file) in enumerate(questions):
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)

            # Format choices consistently with gen_model_answer.py
            choices = all_choices[answer_file][idx]
            for choice in choices:
                # Ensure each turn is properly formatted
                choice["turns"] = [turn.strip() if isinstance(turn, str) else turn for turn in choice["turns"]]

            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "choices": choices,
                "tstamp": time.time(),
            }
            results.append(ans_json)
            with open(os.path.expanduser(answer_file), "a") as fout:
                fout.write(json.dumps(ans_json) + "\n")
            reorg_answer_file(answer_file)

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
        all_results = []
        question_to_date = {}
        if self.question_source == "huggingface":
            categories, tasks = get_categories_tasks(self.dataset_name)

            for category_name, task_names in tasks.items():
                for task_name in task_names:
                    print(f"Evaluating {category_name} {task_name}")
                    questions = load_questions(
                        categories[category_name],
                        self.all_release_dates,
                        self.release_date,
                        task_name,
                    )
                    if self.debug:
                        questions = questions[:10]

                    task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                    output_file = f"{self.data_path}/{task_full_name}/model_judgment/ground_truth_judgment.jsonl"
                    answer_dir = f"{self.data_path}/{task_full_name}/model_answer/"
                    if len(questions) > 0:
                        for question in questions:
                            question_to_date[question["question_id"]] = {
                                "livebench_removal_date": question["livebench_removal_date"],
                                "livebench_release_date": question["livebench_release_date"],
                            }
                        print(f"Judgmenet file wrote: {output_file}")
                        gen_judgments(
                            parallel=self.num_workers,
                            questions=questions,
                            output_file=output_file,
                            answer_dir=answer_dir,
                            model_list=[model_name],
                            remove_existing_file=self.remove_existing_file,
                            bench_name=task_full_name,
                            debug=self.debug,
                        )

                    with open(output_file, "r") as f:
                        all_results.extend([json.loads(line) for line in f])

        elif self.question_source == "jsonl":
            list_of_question_files = []
            original_question_file = f"{self.data_path}/{self.dataset_name}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(
                    f"{self.data_path}/{self.dataset_name}/**/question.jsonl", recursive=True
                )

            for question_file in list_of_question_files:
                # First load the questions
                questions = load_questions_jsonl(question_file, self.all_release_dates, self.release_date)
                if self.debug:
                    questions = questions[:10]

                # Then process them
                for question in questions:
                    question_to_date[question["question_id"]] = {
                        "livebench_removal_date": question["livebench_removal_date"],
                        "livebench_release_date": question["livebench_release_date"],
                    }

                bench_name = os.path.dirname(question_file).replace(f"{self.data_path}/", "")

                output_file = f"{self.data_path}/{bench_name}/model_judgment/ground_truth_judgment.jsonl"
                answer_dir = f"{self.data_path}/{bench_name}/model_answer/"

                if len(questions) > 0:
                    gen_judgments(
                        parallel=self.num_workers,
                        questions=questions,
                        output_file=output_file,
                        answer_dir=answer_dir,
                        model_list=[model_name],
                        remove_existing_file=self.remove_existing_file,
                        bench_name=bench_name,
                    )
                    with open(output_file, "r") as f:
                        all_results.extend([json.loads(line) for line in f])

        else:
            raise ValueError(f"Bad question source {self.question_source}.")

        print("Finished evaluating, calculating metrics")
        # After getting all results, calculate metrics
        metrics = {}
        # Group results by task and calculate averages
        task_scores = {}
        category_scores = {}
        for entry in all_results:
            task = entry["task"]
            category = entry["category"]
            score = entry["score"]
            if task not in task_scores:
                task_scores[task] = []
            task_scores[task].append(score)
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)

        # Calculate task averages
        for task, scores in task_scores.items():
            metrics[f"{task}"] = sum(scores) / len(scores) * 100

        # Calculate category averages
        for category, scores in category_scores.items():
            metrics[f"{category}"] = sum(scores) / len(scores) * 100

        # Calculate global average
        all_scores = [entry["score"] for entry in all_results]
        metrics["global_average"] = sum(all_scores) / len(all_scores) * 100

        # Group by date if timestamp exists
        date_scores = {}
        for entry in all_results:
            date = question_to_date[entry["question_id"]]["livebench_removal_date"]
            if date not in date_scores:
                date_scores[date] = []
            date_scores[date].append(entry["score"])

        # Calculate date averages
        for date, scores in date_scores.items():
            if date == "":
                subset_name = "subset_no_date"
            else:
                subset_name = f"subset_{date}"
            metrics[subset_name] = sum(scores) / len(scores) * 100

        result_dict = {
            "metrics": metrics,
            "num_questions": len(all_results),
        }
        return result_dict

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
            evaluation_results.update({"benchmark_version": "live_bench"})
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
