from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os
import random
import json
import time
import logging
import numpy as np
import pandas as pd
import shortuuid
import torch.distributed as dist
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark
from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    NEED_REF_CATS,
    temperature_config,
)
from fastchat.utils import str_to_torch_dtype
from fastchat.llm_judge.gen_judgment import (
    make_match,
    make_match_all_pairs,
    make_match_single,
    make_judge_single,
    make_judge_pairwise,
)


@dataclass
class MTBenchConfig:
    """Configuration for MTBench evaluation."""

    bench_name: str = "mt_bench"
    question_begin: Optional[int] = None
    question_end: Optional[int] = None
    max_new_token: int = 1024
    num_choices: int = 1
    num_gpus_per_model: int = 1
    num_gpus_total: int = 1
    max_gpu_memory: Optional[str] = None
    dtype: Optional[str] = None
    revision: str = "main"
    judge_file: str = "eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/judge_prompts.jsonl"
    judge_model: str = "gpt-4o-mini-2024-07-18"
    baseline_model: str = "gpt-3.5-turbo"
    mode: str = "single"
    parallel: int = 4
    first_n: Optional[int] = None


class MTBenchBenchmark(BaseBenchmark):
    """
    MTBench benchmark for evaluating multi-turn chat capabilities.
    """

    def __init__(
        self,
        base_path: str = "eval/chat_benchmarks/MTBench",
        config: Optional[MTBenchConfig] = None,
        debug: bool = False,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize MTBench benchmark.

        Args:
            base_path: Base directory for MTBench data and outputs
            config: MTBench configuration object
            debug: If True, run in debug mode on 2 samples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.base_path = Path(base_path)
        if annotator_model == "auto":
            annotator_model = "gpt-4"
        if config:
            print(f"Warning: Overwriting config.judge_model = {annotator_model} ")
            config.judge_model = annotator_model
        self.config = config or MTBenchConfig(judge_model=annotator_model)
        self.debug = debug

        # Setup paths
        self.data_path = self.base_path / "fastchat/llm_judge/data/mt_bench"
        self.question_file = self.data_path / "question.jsonl"
        self.answer_dir = self.data_path / "model_answer"
        self.ref_answer_dir = self.data_path / "reference_answer"
        self.judgment_dir = self.data_path / "model_judgment"

        # Create directories
        self.answer_dir.mkdir(parents=True, exist_ok=True)
        self.judgment_dir.mkdir(parents=True, exist_ok=True)

    def get_model_answers(self, model: LM, model_id: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate model answers for all questions."""
        # Initialize tracking structures
        all_convs = [[] for _ in questions]
        all_choices = [{"index": 0, "turns": []} for _ in questions]

        max_turns = max(len(q["turns"]) for q in questions)
        answer_file = self.answer_dir / f"{model_id}.jsonl"

        # Process each turn
        for turn_num in range(max_turns):
            self.logger.info(f"Processing Turn {turn_num + 1}")
            batch_instances = []

            # Prepare instances for current turn
            self.logger.info("Generating responses for MTBench...")
            for q_idx, question in enumerate(questions):
                if turn_num < len(question["turns"]):
                    temperature = temperature_config.get(question["category"], 0.7)

                    # Add user message to conversation
                    all_convs[q_idx].append({"role": "user", "content": question["turns"][turn_num]})

                    # Prepare model input
                    prompt = model.apply_chat_template(all_convs[q_idx])
                    batch_instances.append(
                        Instance(
                            "generate_until",
                            all_convs[q_idx],
                            (
                                prompt,
                                {
                                    "max_gen_toks": self.config.max_new_token,
                                    "do_sample": temperature >= 1e-4,
                                    "temperature": temperature,
                                },
                            ),
                            q_idx,
                        )
                    )

            # Generate responses
            if batch_instances:
                outputs = self.compute(model, batch_instances)

                # Process outputs
                for q_idx, output in enumerate(outputs):
                    all_convs[q_idx].append({"role": "assistant", "content": output})
                    all_choices[q_idx]["turns"].append(output)

            is_main_process = model.accelerator.process_index == 0 if hasattr(model, 'accelerator') else model.world_size <= 1
            if not is_main_process:
                continue

            # Save completed conversations
            for q_idx, question in enumerate(questions):
                if turn_num == len(question["turns"]) - 1:
                    ans_json = {
                        "question_id": question["question_id"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_id,
                        "choices": [all_choices[q_idx]],
                        "tstamp": time.time(),
                    }
                    with open(answer_file, "a") as f:
                        f.write(json.dumps(ans_json) + "\n")

        return all_choices

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for MTBench questions.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing model identifier, or None for non-primary ranks
        """
        # Load questions
        questions = load_questions(self.question_file, self.config.question_begin, self.config.question_end)

        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode: using first 2 questions")

        # Shuffle questions for better load balancing
        random.shuffle(questions)

        # Generate answers
        answers = self.get_model_answers(model=model, model_id=model.model_identifier, questions=questions)

        # Return None early for non-primary ranks if compute() returned None
        if answers is None:
            return None

        return {"model_id": model.model_identifier}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model responses using GPT-4 judge.

        Args:
            results: Dictionary containing model identifier

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        # Load data
        questions = load_questions(self.question_file, None, None)
        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode: using 2 examples")

        model_answers = load_model_answers(self.answer_dir)
        ref_answers = load_model_answers(self.ref_answer_dir)
        judge_prompts = load_judge_prompts(self.config.judge_file)

        # Setup evaluation
        models = [results["model_id"]]
        if self.config.mode == "single":
            judges = make_judge_single(self.config.judge_model, judge_prompts)
            play_a_match_func = play_a_match_single
            output_file = self.judgment_dir / "gpt-4_single.jsonl"
            make_match_func = make_match_single
            baseline_model = None
        else:
            judges = make_judge_pairwise(self.config.judge_model, judge_prompts)
            play_a_match_func = play_a_match_pair
            output_file = self.judgment_dir / "gpt-4_pair.jsonl"
            if self.config.mode == "pairwise-all":
                make_match_func = make_match_all_pairs
                baseline_model = None
            else:
                make_match_func = make_match
                baseline_model = self.config.baseline_model

        # Verify data
        check_data(questions, model_answers, ref_answers, models, judges)

        # Split questions by category
        questions_math = [q for q in questions if q["category"] in NEED_REF_CATS]
        questions_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

        # Create matches
        matches = []
        matches.extend(make_match_func(questions_default, models, model_answers, judges["default"], baseline_model))
        matches.extend(
            make_match_func(
                questions_math,
                models,
                model_answers,
                judges["math"],
                baseline_model,
                ref_answers,
            )
        )
        matches.extend(
            make_match_func(
                questions_default,
                models,
                model_answers,
                judges["default-mt"],
                baseline_model,
                multi_turn=True,
            )
        )
        matches.extend(
            make_match_func(
                questions_math,
                models,
                model_answers,
                judges["math-mt"],
                baseline_model,
                ref_answers,
                multi_turn=True,
            )
        )

        # Run evaluation
        if self.config.parallel == 1:
            for match in tqdm(matches):
                play_a_match_func(match, output_file=output_file)
        else:
            with ThreadPoolExecutor(self.config.parallel) as executor:
                list(
                    tqdm(
                        executor.map(lambda m: play_a_match_func(m, output_file=output_file), matches),
                        total=len(matches),
                    )
                )

        # Load and process results
        df_all = pd.read_json(output_file, lines=True)
        df = df_all[["model", "score", "turn"]]
        df = df[df["score"] != -1]

        # Calculate scores
        df_turn1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
        df_turn2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        df_avg = df[["model", "score"]].groupby(["model"]).mean()

        model_id = results["model_id"]
        return {
            "Turn 1": df_turn1.loc[results["model_id"]].score.values[0],
            "Turn 2": df_turn2.loc[results["model_id"]].score.values[0],
            "Average": df_avg.loc[results["model_id"]].score,
        }

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete MTBench evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting MTBench evaluation")
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
