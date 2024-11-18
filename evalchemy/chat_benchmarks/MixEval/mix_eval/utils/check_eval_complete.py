"""
Usage:
python -m mix_eval.utils.check_eval_complete \
    --benchmark {mixeval, mixeval_hard} \
    --version VERSION \
    --base_models_to_check BASE_MODELS_TO_CHECK [BASE_MODELS_TO_CHECK ...] \
    --chat_models_to_check CHAT_MODELS_TO_CHECK [CHAT_MODELS_TO_CHECK ...] \
    [--n_closefreeform N_CLOSEFREEFORM] \
    [--n_closemultichoice N_CLOSEMULTICHOICE] \
    [--model_response_dir RESPONSE_DIR] \
    [--out_path OUT_PATH]
"""

import json
import argparse
import os
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from mix_eval.models import AVAILABLE_MODELS
from mix_eval.utils.common_utils import log_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, required=True, choices=["mixeval", "mixeval_hard"], help="Benchmark to evaluate."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="The benchmark version to run. We update MixEval data points on a monthly basis.",
    )
    parser.add_argument(
        "--base_models_to_check", nargs="+", required=True, help="Base models to check. Set to None if not needed."
    )
    parser.add_argument(
        "--chat_models_to_check", nargs="+", required=True, help="Base models to check. Set to None if not needed."
    )
    parser.add_argument("--n_closefreeform", type=int, default=2000, help="Valid size for close freeform split.")
    parser.add_argument("--n_closemultichoice", type=int, default=2000, help="Valid size for close multi-choice split.")
    parser.add_argument(
        "--n_closefreeform_hard", type=int, default=500, help="Valid size for close freeform-hard split."
    )
    parser.add_argument(
        "--n_closemultichoice_hard", type=int, default=500, help="Valid size for close multi-choice-hard split."
    )
    parser.add_argument(
        "--model_response_dir", default="mix_eval/data/model_responses", type=str, help="The model response directory."
    )
    parser.add_argument(
        "--out_path",
        default="mix_eval/data/model_responses/eval_checks.log",
        type=str,
        help="The check file to write to.",
    )
    return parser.parse_args()


def check_result(args, model_dir, correct_num, split, model_name):
    status_complete = True
    num_correct = True
    no_error = True

    if not os.path.exists(model_dir):
        message = f"Directory {model_dir} does not exist."
        log_error(message, args.out_path)
        return

    # check status
    status_file_path = os.path.join(model_dir, f"status_{split}.json")
    if not os.path.exists(status_file_path):
        print(status_file_path)
        status_complete = False
    else:
        with open(status_file_path, "r") as f:
            status = json.load(f)
        if status["status"]["status"] != "complete":
            status_complete = False

    # check number of responses
    response_file_path = os.path.join(model_dir, f"{model_name}_{split}.jsonl")
    if not os.path.exists(response_file_path):
        num_correct = False
    else:
        with open(response_file_path, "r") as f:
            lines = f.readlines()
            response_num = len(lines)
            num_correct = response_num == correct_num

    # check error
    # log_file_path = os.path.join(
    #         model_dir,
    #         f"{os.path.basename(model_dir)}.log"
    #         )
    # if not os.path.exists(log_file_path):
    #     no_error = False
    # else:
    #     with open(
    #         log_file_path,
    #         "r"
    #         ) as f:
    #         logfile = f.read().lower()
    #         if "error" in logfile:
    #             no_error = False

    if status_complete and num_correct and no_error:
        pass
    else:
        message = (
            f"Directory {model_dir} has issues. Please check the log file inside. "
            f"The check result: [Status complete: {status_complete}], "
            f"[Responses file complete: {num_correct}], "
            f"[No error in log file: {no_error}]."
        )
        log_error(message, args.out_path)


def check_results_base(args, base_models_to_check):
    if args.benchmark == "mixeval":
        split = "close_freeform"
        for model in base_models_to_check:
            model_dir = os.path.join(args.model_response_dir, model, args.benchmark, args.version)
            check_result(args, model_dir, args.n_closefreeform, split, model)

        split = "close_multichoice"
        for model in base_models_to_check:
            model_dir = os.path.join(args.model_response_dir, model, args.benchmark, args.version)
            check_result(args, model_dir, args.n_closemultichoice, split, model)

    elif args.benchmark == "mixeval_hard":
        split = "close_freeform_hard"
        for model in base_models_to_check:
            model_dir = os.path.join(args.model_response_dir, model, args.benchmark, args.version)
            check_result(args, model_dir, args.n_closefreeform_hard, split, model)

        split = "close_multichoice_hard"
        for model in base_models_to_check:
            model_dir = os.path.join(args.model_response_dir, model, args.benchmark, args.version)
            check_result(args, model_dir, args.n_closemultichoice_hard, split, model)


def check_results_chat(args, chat_models_to_check):
    if args.benchmark == "mixeval":
        split = "close_freeform"
        for model in chat_models_to_check:
            model_dir = os.path.join(args.model_response_dir, model, args.benchmark, args.version)
            check_result(args, model_dir, args.n_closefreeform, split, model)

        split = "close_multichoice"
        for model in chat_models_to_check:
            model_dir = os.path.join(args.model_response_dir, model, args.benchmark, args.version)
            check_result(args, model_dir, args.n_closemultichoice, split, model)
    elif args.benchmark == "mixeval_hard":
        split = "close_freeform_hard"
        for model in chat_models_to_check:
            model_dir = os.path.join(args.model_response_dir, model, args.benchmark, args.version)
            check_result(args, model_dir, args.n_closefreeform_hard, split, model)

        split = "close_multichoice_hard"
        for model in chat_models_to_check:
            model_dir = os.path.join(args.model_response_dir, model, args.benchmark, args.version)
            check_result(args, model_dir, args.n_closemultichoice_hard, split, model)
    else:
        raise ValueError(f"Benchmark {args.benchmark} not supported.")


def check_results(args, base_models_to_check, chat_models_to_check):
    base_models_to_check = [m for m in base_models_to_check if m.lower() != "none"]
    chat_models_to_check = [m for m in chat_models_to_check if m.lower() != "none"]
    for m_b in base_models_to_check:
        if m_b.lower() != "None":
            assert m_b in AVAILABLE_MODELS.keys(), f"Model {m_b} not supported."
    for m_c in chat_models_to_check:
        if m_c.lower() != "None":
            assert m_c in AVAILABLE_MODELS.keys(), f"Model {m_c} not supported."

    check_results_base(args, base_models_to_check)
    check_results_chat(args, chat_models_to_check)

    message = f"Above lines are problematic directories. No lines above means all entries are valid. "
    log_error(message, args.out_path)


if __name__ == "__main__":
    args = parse_args()
    model_b = args.base_models_to_check
    model_c = args.chat_models_to_check
    check_results(args, model_b, model_c)
