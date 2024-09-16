from dataclasses import dataclass
import os
import random
from tqdm import tqdm
import pandas as pd
import shortuuid
import time
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor

from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
    temperature_config,
)
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from fastchat.llm_judge.gen_judgment import make_match, make_match_all_pairs, make_match_single, make_judge_single

from lm_eval.api.instance import Instance

from lm_eval.utils import eval_logger
from typing import List, Optional


@dataclass
class ShowResultsConfig:
    """Configuration for the evaluation script."""

    bench_name: str = "mt_bench"
    """Name of the benchmark."""

    input_file: str = None
    """Path to the input file."""

    judge_model: str = "gpt-4"
    """Model used for judging."""

    baseline_model: str = "gpt-3.5-turbo"
    """Baseline model for comparison."""

    model_list: Optional[List[str]] = None
    """List of models to be evaluated."""

    mode: str = "single"
    """Evaluation mode. Can be 'pairwise-baseline', 'pairwise-all', or 'single'."""


@dataclass
class EvaluateInstructConfig:
    bench_name: str = "mt_bench"
    question_begin: int = None
    question_end: int = None
    answer_file: str = None
    max_new_token: int = 1024
    num_choices: int = 1
    num_gpus_per_model: int = 1
    num_gpus_total: int = 1
    max_gpu_memory: str = None
    dtype: str = None
    revision: str = "main"


@dataclass
class EvaluateConfig:
    bench_name: str = "mt_bench"
    judge_file: str = "eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/judge_prompts.jsonl"
    judge_model: str = "gpt-4"
    baseline_model: str = "gpt-3.5-turbo"
    mode: str = "single"
    model_list: list[str] = None
    parallel: int = 4
    first_n: int = None


def run_eval(
    model,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    get_model_answers(
        model,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        dtype=dtype,
        revision=revision,
    )


def get_model_answers(
    model,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
):
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    # Initialize structures to keep track of conversations and choices
    all_convs = [[] for _ in questions]
    all_choices = [{"index": 0, "turns": []} for _ in questions]

    max_turns = max(len(q["turns"]) for q in questions)

    for turn_num in range(max_turns):
        eval_logger.info(f"Processing Turn {turn_num + 1}")
        batch_instances = []

        for q_idx, question in enumerate(tqdm(questions)):
            if turn_num < len(question["turns"]):
                temperature = temperature_config.get(question["category"], 0.7)
                do_sample = temperature >= 1e-4

                all_convs[q_idx].append({"role": "user", "content": question["turns"][turn_num]})
                conv = all_convs[q_idx]

                prompt = model.apply_chat_template(conv)
                batch_instances.append(
                    Instance(
                        "generate_until",
                        conv,
                        (
                            prompt,
                            {"max_new_tokens": max_new_token, "do_sample": do_sample, "temperature": temperature},
                        ),
                        q_idx,
                    )
                )

        if batch_instances:
            outputs = model.generate_until(batch_instances)

            for q_idx in range(len(outputs)):
                output = outputs[q_idx]
                all_convs[q_idx].append({"role": "assistant", "content": output})
                all_choices[q_idx]["turns"].append(output)

        # Write completed questions to file
        for q_idx, question in enumerate(questions):
            if turn_num == len(question["turns"]) - 1:
                with open(os.path.expanduser(answer_file), "a") as fout:
                    ans_json = {
                        "question_id": question["question_id"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_id,
                        "choices": [all_choices[q_idx]],
                        "tstamp": time.time(),
                    }
                    fout.write(json.dumps(ans_json) + "\n")

    eval_logger.info(f"Completed processing all questions. Results written to {answer_file}")


def eval_instruct(model):

    question_file = f"eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/mt_bench/question.jsonl"

    config = EvaluateInstructConfig()
    model_id = model.model_identifier

    answer_file = f"/import/ml-sc-scratch5/etashg/dcft_private/eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/mt_bench/model_answer/{model_id}.jsonl"

    eval_logger.info(f"Output to {answer_file}")
    run_eval(
        model=model,
        model_id=model.model_identifier,
        question_file=question_file,
        question_begin=config.question_begin,
        question_end=config.question_end,
        answer_file=answer_file,
        max_new_token=config.max_new_token,
        num_choices=config.num_choices,
        num_gpus_per_model=config.num_gpus_per_model,
        num_gpus_total=config.num_gpus_total,
        max_gpu_memory=config.max_gpu_memory,
        dtype=str_to_torch_dtype(config.dtype),
        revision=config.revision,
    )
    results = {}
    results["model_id"] = model.model_identifier
    return results


def evaluate(results):
    config = EvaluateConfig()

    question_file = f"eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/mt_bench/question.jsonl"
    answer_dir = f"eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/mt_bench/model_answer"
    ref_answer_dir = f"eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/mt_bench/reference_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)
    # Load judge
    judge_prompts = load_judge_prompts(config.judge_file)

    if config.first_n:
        questions = questions[: config.first_n]

    models = [results["model_id"]]
    if config.mode == "single":
        judges = make_judge_single(config.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = f"eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_single.jsonl"
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(config.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = f"eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_pair.jsonl"
        if config.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = config.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(question_default, models, model_answers, judges["default"], baseline_model)
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = config.bench_name
    match_stat["mode"] = config.mode
    match_stat["judge"] = config.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    # Play matches
    if config.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.shuffle(matches)

        with ThreadPoolExecutor(config.parallel) as executor:
            for match in tqdm(executor.map(play_a_match_wrapper, matches), total=len(matches)):
                pass

    df_all = pd.read_json(
        f"eval/chat_benchmarks/MTBench/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_single.jsonl", lines=True
    )

    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    print("\n########## Second turn ##########")
    df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
    print(df_2.sort_values(by="score", ascending=False))

    print("\n########## Average ##########")
    df_3 = df[["model", "score"]].groupby(["model"]).mean()
    print(df_3.sort_values(by="score", ascending=False))
    results = {
        "Turn 1": df_1.loc[results["model_id"]].score.values[0],
        "Turn 2": df_2.loc[results["model_id"]].score.values[0],
        "Average": df_3.loc[results["model_id"]].score,
    }
    return results
