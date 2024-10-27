"""
Usage:
python -m mix_eval.compute_metrics \
    --benchmark {mixeval, mixeval_hard} \
    --version VERSION \
    --models_to_eval MODELS_TO_EVAL [MODELS_TO_EVAL ...] \
    [--model_response_dir RESULTS_DIR] \
    [--multichoice_judge MULTICHOICE_JUDGE] \ 
    [--freeform_judge FREEFORM_JUDGE] \ 
    [--free_form_parser {model,rule}] \ 
    [--multi_choice_parser {model,rule}] \
    [--api_parallel_num API_PARALLEL_NUM] \ 
    [--extract_base_model_response] \ 
    [--compute_score_from_judged_file]
"""

import json
import argparse
import os
from tqdm import tqdm
import time
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
from prettytable import PrettyTable

import mix_eval.api.registry
from mix_eval.utils.common_utils import set_seed
from mix_eval.utils.metric_utils import (
    parse_multi_choice_response_rule,
    parse_multi_choice_response_model,
    eval_multi_choice,
    eval_freeform_model,
    parse_freeform_response_rule,
    eval_freeform_rule,
)
from mix_eval.models import AVAILABLE_MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", type=str, choices=["mixeval", "mixeval_hard"], required=True, help="Benchmark to evaluate."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="The benchmark version to run. We update MixEval data points on a monthly basis.",
    )
    parser.add_argument(
        "--model_response_dir", type=str, default="mix_eval/data/model_responses/", help="Path to model responses."
    )
    parser.add_argument(
        "--multichoice_judge",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="Judge model for multiple-choice score computation.",
    )
    parser.add_argument(
        "--freeform_judge", type=str, default="gpt-3.5-turbo-0125", help="Judge model for freeform score computation."
    )
    parser.add_argument("--models_to_eval", nargs="+", default=None, help="Models to evaluate.")
    parser.add_argument(
        "--free_form_parser",
        type=str,
        default="model",
        choices=["model", "rule"],
        help="Parser for freeform responses, either model parser or rule-based parser.",
    )
    parser.add_argument(
        "--multi_choice_parser",
        type=str,
        default="model",
        choices=["model", "rule"],
        help="Parser for multiple-choice responses, either model parser or rule-based parser.",
    )
    parser.add_argument("--api_base_url", type=str, default=None, help="The base url for the model parser api.")
    parser.add_argument(
        "--api_parallel_num",
        type=int,
        default=100,
        help="Number of parallel threads for calling the model parser api if use model parsing."
        "If you hit rate limit error frequently, try to reduce this number.",
    )
    parser.add_argument(
        "--extract_base_model_response",
        action="store_true",
        help="The unfinetuned models will produce endless output, " "which may influence the model parse score.",
    )
    parser.add_argument(
        "--compute_score_from_judged_file",
        action="store_true",
        help="Whether to compute score directly from the judged file."
        "This will save budge for those models that has been judged before."
        "it also helps to do some analysis easily without running judgements again.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose information.")
    return parser.parse_args()


def compute_metric_closeended_freeform_modelparse_from_judgefile(args):
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        for model in models:
            if not model in AVAILABLE_MODELS.keys():
                print(f"Model {model} is not available in the registry.")
        models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
            for model in models:
                if not model in AVAILABLE_MODELS.keys():
                    print(f"Model {model} is not available in the registry.")
            models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    for model in models:
        print(f"Parsing model: {model}")
        score_dict_model = {}
        judge_file = os.path.join(
            args.model_response_dir,
            model,
            args.benchmark,
            args.version,
            f"judge_results_ff_model_judge_{args.freeform_judge}.jsonl",
        )
        if not os.path.exists(judge_file):
            print(f"Judge file not found: {judge_file}")
            continue
        with open(judge_file, "r") as f:
            for line in f:
                judge_dict = json.loads(line)
                judge_score = judge_dict["judge_score"]
                if "overall" not in score_dict_model:
                    score_dict_model["overall"] = []
                score_dict_model["overall"].append(judge_score)
                if judge_dict["benchmark_name"] not in score_dict_model:
                    score_dict_model[judge_dict["benchmark_name"]] = []
                score_dict_model[judge_dict["benchmark_name"]].append(judge_score)

        score_dict_counts = {}

        for key, value in score_dict_model.items():
            score_dict_counts[key] = len(value)
            score_dict_model[key] = round(sum(value) / len(value), 3)

        score_dict[model] = score_dict_model
        score_dict[model]["number_samples"] = score_dict_counts
    return score_dict


def compute_metric_closeended_freeform_ruleparse_from_judgefile(args):
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        for model in models:
            if not model in AVAILABLE_MODELS.keys():
                print(f"Model {model} is not available in the registry.")
        models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
            for model in models:
                if not model in AVAILABLE_MODELS.keys():
                    print(f"Model {model} is not available in the registry.")
            models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    for model in models:
        print(f"Parsing model: {model}")
        score_dict_model = {}
        judge_file = os.path.join(
            args.model_response_dir, model, args.benchmark, args.version, f"judge_results_ff_rule.jsonl"
        )
        if not os.path.exists(judge_file):
            print(f"Judge file not found: {judge_file}")
            continue
        with open(judge_file, "r") as f:
            for line in f:
                judge_dict = json.loads(line)
                judge_score = 1 if judge_dict["eval_result"] else 0
                if "overall" not in score_dict_model:
                    score_dict_model["overall"] = []
                score_dict_model["overall"].append(judge_score)
                if judge_dict["benchmark_name"] not in score_dict_model:
                    score_dict_model[judge_dict["benchmark_name"]] = []
                score_dict_model[judge_dict["benchmark_name"]].append(judge_score)

        score_dict_counts = {}

        for key, value in score_dict_model.items():
            score_dict_counts[key] = len(value)
            score_dict_model[key] = round(sum(value) / len(value), 3)

        score_dict[model] = score_dict_model
        score_dict[model]["number_samples"] = score_dict_counts

    return score_dict


def compute_metric_closeended_multichoice_modelparse_from_judgefile(args):
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        for model in models:
            if not model in AVAILABLE_MODELS.keys():
                print(f"Model {model} is not available in the registry.")
        models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
            for model in models:
                if not model in AVAILABLE_MODELS.keys():
                    print(f"Model {model} is not available in the registry.")
            models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    for model in models:
        print(f"Parsing model: {model}")
        score_dict_model = {}
        judge_file = os.path.join(
            args.model_response_dir,
            model,
            args.benchmark,
            args.version,
            f"judge_results_mp_model_judge_{args.multichoice_judge}.jsonl",
        )
        if not os.path.exists(judge_file):
            print(f"Judge file not found: {judge_file}")
            continue
        with open(judge_file, "r") as f:
            for line in f:
                # compute score
                judge_dict = json.loads(line)
                options = judge_dict["options"]
                target = judge_dict["target"]
                assert isinstance(target, list) and len(target) == 1, f"Invalid target: {target}"
                all_choices = [chr(ord("A") + i) for i in range(len(options))]
                model_choice = judge_dict["judge_option"]
                target_id = all_choices[target[0]]
                judge_score = 1 if eval_multi_choice(target_id, model_choice) else 0

                # add score
                if "overall" not in score_dict_model:
                    score_dict_model["overall"] = []
                score_dict_model["overall"].append(judge_score)
                if judge_dict["benchmark_name"] not in score_dict_model:
                    score_dict_model[judge_dict["benchmark_name"]] = []
                score_dict_model[judge_dict["benchmark_name"]].append(judge_score)

        score_dict_counts = {}

        for key, value in score_dict_model.items():
            score_dict_counts[key] = len(value)
            score_dict_model[key] = round(sum(value) / len(value), 3)

        score_dict[model] = score_dict_model
        score_dict[model]["number_samples"] = score_dict_counts

    return score_dict


def compute_metric_closeended_multichoice_ruleparse_from_judgefile(args):
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        for model in models:
            if not model in AVAILABLE_MODELS.keys():
                print(f"Model {model} is not available in the registry.")
        models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
            for model in models:
                if not model in AVAILABLE_MODELS.keys():
                    print(f"Model {model} is not available in the registry.")
            models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    for model in models:
        print(f"Parsing model: {model}")
        score_dict_model = {}
        judge_file = os.path.join(
            args.model_response_dir, model, args.benchmark, args.version, f"judge_results_mp_rule.jsonl"
        )
        if not os.path.exists(judge_file):
            print(f"Judge file not found: {judge_file}")
            continue
        with open(judge_file, "r") as f:
            for line in f:
                # compute score
                judge_dict = json.loads(line)
                judge_score = 1 if judge_dict["eval_result"] else 0

                # add score
                if "overall" not in score_dict_model:
                    score_dict_model["overall"] = []
                score_dict_model["overall"].append(judge_score)
                if judge_dict["benchmark_name"] not in score_dict_model:
                    score_dict_model[judge_dict["benchmark_name"]] = []
                score_dict_model[judge_dict["benchmark_name"]].append(judge_score)

        score_dict_counts = {}

        for key, value in score_dict_model.items():
            score_dict_counts[key] = len(value)
            score_dict_model[key] = round(sum(value) / len(value), 3)

        score_dict[model] = score_dict_model
        score_dict[model]["number_samples"] = score_dict_counts

    return score_dict


def compute_metric_closeended_freeform_modelparse(args):
    if args.compute_score_from_judged_file:
        return compute_metric_closeended_freeform_modelparse_from_judgefile(args)

    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
        else:
            raise ValueError(f"Model response directory not found: {args.model_response_dir}")

    for model in models:
        print(f"\n\n\nParsing model: {model}\n\n\n")
        if args.extract_base_model_response:
            try:
                args.model_type = mix_eval.api.registry.get_model(model).__bases__[0].__name__
            except ValueError as e:
                print(f"Model {model} is not available in the registry.")
                args.model_type = "hf"

        if args.benchmark == "mixeval":
            split = "close_freeform"
        elif args.benchmark == "mixeval_hard":
            split = "close_freeform_hard"
        else:
            raise ValueError(f"Invalid benchmark: {args.benchmark}.")
        ans_file = os.path.join(args.model_response_dir, model, args.benchmark, args.version, f"{model}_{split}.jsonl")
        tasks = []
        with open(ans_file, "r") as f:
            for line in f:
                ans_dict = json.loads(line)
                tasks.append(ans_dict)
        results = eval_freeform_model(args, tasks)

        score_dict_model = {}
        for judge_dict in results:
            judge_score = judge_dict["judge_score"]
            if "overall" not in score_dict_model:
                score_dict_model["overall"] = []
            score_dict_model["overall"].append(judge_score)
            if judge_dict["benchmark_name"] not in score_dict_model:
                score_dict_model[judge_dict["benchmark_name"]] = []
            score_dict_model[judge_dict["benchmark_name"]].append(judge_score)

        score_dict_counts = {}

        for key, value in score_dict_model.items():
            score_dict_counts[key] = len(value)
            score_dict_model[key] = round(sum(value) / len(value), 3)

        score_dict[model] = score_dict_model
        score_dict[model]["number_samples"] = score_dict_counts

        with open(
            os.path.join(
                args.model_response_dir,
                model,
                args.benchmark,
                args.version,
                f"judge_results_ff_model_judge_{args.freeform_judge}.jsonl",
            ),
            "w",
        ) as f:
            for case in results:
                f.write(json.dumps(case) + "\n")

        print("Sleep 10 seconds to avoid ratelimit error ... ")
        time.sleep(10)

    if args.verbose:
        print(f"[Close-ended Free-form Model Parser]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")

    return score_dict


def compute_metric_closeended_freeform_ruleparse(args):
    if args.compute_score_from_judged_file:
        return compute_metric_closeended_freeform_ruleparse_from_judgefile(args)

    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        for model in models:
            if not model in AVAILABLE_MODELS.keys():
                print(f"Model {model} is not available in the registry.")
        models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
            for model in models:
                if not model in AVAILABLE_MODELS.keys():
                    print(f"Model {model} is not available in the registry.")
            models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    for model in tqdm(models):
        if args.extract_base_model_response:
            args.model_type = mix_eval.api.registry.get_model(model).__bases__[0].__name__
        if args.benchmark == "mixeval":
            split = "close_freeform"
        elif args.benchmark == "mixeval_hard":
            split = "close_freeform_hard"
        else:
            raise ValueError(f"Invalid benchmark: {args.benchmark}.")
        ans_file = os.path.join(args.model_response_dir, model, args.benchmark, args.version, f"{model}_{split}.jsonl")
        with open(ans_file, "r") as f:
            total = 0
            correct = 0
            results = []
            error_cases = []
            for line in f:
                ans_dict = json.loads(line)
                target = ans_dict["target"]
                assert isinstance(target, list)
                model_response = ans_dict["response"]
                preds = parse_freeform_response_rule(args, model_response)
                if eval_freeform_rule(target, preds):
                    correct += 1
                else:
                    error_cases.append(ans_dict)
                ans_dict["parsed_result"] = preds
                ans_dict["eval_result"] = eval_freeform_rule(target, preds)
                results.append(ans_dict)
                total += 1

        score_dict_model = {}
        for judge_dict in results:
            judge_score = 1 if judge_dict["eval_result"] else 0
            if "overall" not in score_dict_model:
                score_dict_model["overall"] = []
            score_dict_model["overall"].append(judge_score)
            if judge_dict["benchmark_name"] not in score_dict_model:
                score_dict_model[judge_dict["benchmark_name"]] = []
            score_dict_model[judge_dict["benchmark_name"]].append(judge_score)

        score_dict_counts = {}

        for key, value in score_dict_model.items():
            score_dict_counts[key] = len(value)
            score_dict_model[key] = round(sum(value) / len(value), 3)

        score_dict[model] = score_dict_model
        score_dict[model]["number_samples"] = score_dict_counts

        with open(
            os.path.join(args.model_response_dir, model, args.benchmark, args.version, "judge_results_ff_rule.jsonl"),
            "w",
        ) as f:
            for case in results:
                f.write(json.dumps(case) + "\n")
    if args.verbose:
        print(f"[Close-ended Free-form Rule Parser]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")

    return score_dict


def compute_metric_closeended_freeform(args):
    if args.free_form_parser == "model":
        return compute_metric_closeended_freeform_modelparse(args)
    else:
        return compute_metric_closeended_freeform_ruleparse(args)


def compute_metric_closeended_multichoice_modelparse(args):
    if args.compute_score_from_judged_file:
        return compute_metric_closeended_multichoice_modelparse_from_judgefile(args)

    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval

    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)

        else:
            raise ValueError(f"Model response directory not found: {args.model_response_dir}")

    for model in models:
        print(f"\n\n\nParsing model: {model}\n\n\n")
        if args.extract_base_model_response:
            args.model_type = mix_eval.api.registry.get_model(model).__bases__[0].__name__
        if args.benchmark == "mixeval":
            split = "close_multichoice"
        elif args.benchmark == "mixeval_hard":
            split = "close_multichoice_hard"
        else:
            raise ValueError(f"Invalid benchmark: {args.benchmark}.")
        ans_file = os.path.join(args.model_response_dir, model, args.benchmark, args.version, f"{model}_{split}.jsonl")
        with open(ans_file, "r") as f:
            ans_dicts = []
            for line in f:
                ans_dict = json.loads(line)
                ans_dicts.append(ans_dict)

            ans_dicts_withscore = parse_multi_choice_response_model(args, ans_dicts)

            total = 0
            correct = 0
            results = []
            error_cases = []
            for ans_dict_ws in ans_dicts_withscore:
                options = ans_dict_ws["options"]
                target = ans_dict_ws["target"]
                assert isinstance(target, list) and len(target) == 1, f"Invalid target: {target}"
                all_choices = [chr(ord("A") + i) for i in range(len(options))]
                model_choice = ans_dict_ws["judge_option"]
                target_id = all_choices[target[0]]
                if eval_multi_choice(target_id, model_choice):
                    correct += 1
                else:
                    error_cases.append(ans_dict_ws)
                results.append(ans_dict_ws)
                total += 1

        score_dict_model = {}
        for judge_dict in results:
            options = judge_dict["options"]
            target = judge_dict["target"]
            assert isinstance(target, list) and len(target) == 1, f"Invalid target: {target}"
            all_choices = [chr(ord("A") + i) for i in range(len(options))]
            model_choice = judge_dict["judge_option"]
            target_id = all_choices[target[0]]
            judge_score = 1 if eval_multi_choice(target_id, model_choice) else 0

            # add score
            if "overall" not in score_dict_model:
                score_dict_model["overall"] = []
            score_dict_model["overall"].append(judge_score)
            if judge_dict["benchmark_name"] not in score_dict_model:
                score_dict_model[judge_dict["benchmark_name"]] = []
            score_dict_model[judge_dict["benchmark_name"]].append(judge_score)

        score_dict_counts = {}

        for key, value in score_dict_model.items():
            score_dict_counts[key] = len(value)
            score_dict_model[key] = round(sum(value) / len(value), 3)

        score_dict[model] = score_dict_model
        score_dict[model]["number_samples"] = score_dict_counts

        with open(
            os.path.join(
                args.model_response_dir,
                model,
                args.benchmark,
                args.version,
                f"judge_results_mp_model_judge_{args.multichoice_judge}.jsonl",
            ),
            "w",
        ) as f:
            for case in results:
                f.write(json.dumps(case) + "\n")

        print("Sleep 10 seconds to avoid ratelimit error ... ")
        time.sleep(10)

    if args.verbose:
        print(f"[Close-ended Multiple-choice Model Parser]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")

    return score_dict


def compute_metric_closeended_multichoice_ruleparse(args):
    if args.compute_score_from_judged_file:
        return compute_metric_closeended_multichoice_ruleparse_from_judgefile(args)

    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        for model in models:
            if not model in AVAILABLE_MODELS.keys():
                print(f"Model {model} is not available in the registry.")
        models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
            for model in models:
                if not model in AVAILABLE_MODELS.keys():
                    print(f"Model {model} is not available in the registry.")
            models = [model for model in models if model in AVAILABLE_MODELS.keys()]

    for model in tqdm(models):
        if args.extract_base_model_response:
            args.model_type = mix_eval.api.registry.get_model(model).__bases__[0].__name__
        if args.benchmark == "mixeval":
            split = "close_multichoice"
        elif args.benchmark == "mixeval_hard":
            split = "close_multichoice_hard"
        else:
            raise ValueError(f"Invalid benchmark: {args.benchmark}.")
        ans_file = os.path.join(args.model_response_dir, model, args.benchmark, args.version, f"{model}_{split}.jsonl")
        with open(ans_file, "r") as f:
            total = 0
            correct = 0
            results = []
            error_cases = []
            for line in f:
                ans_dict = json.loads(line)
                options = ans_dict["options"]
                target = ans_dict["target"]
                assert isinstance(target, list) and len(target) == 1, f"Invalid target: {target}"
                model_response = ans_dict["response"]
                all_choices = [chr(ord("A") + i) for i in range(len(options))]
                index2ans = {id: option for id, option in zip(all_choices, options)}
                model_choice = parse_multi_choice_response_rule(args, model_response, all_choices, index2ans)
                target_id = all_choices[target[0]]
                if eval_multi_choice(target_id, model_choice):
                    correct += 1
                else:
                    error_cases.append(ans_dict)
                ans_dict["parsed_result"] = model_choice
                ans_dict["eval_result"] = eval_multi_choice(target_id, model_choice)
                results.append(ans_dict)
                total += 1

        score_dict_model = {}
        for judge_dict in results:
            judge_score = 1 if judge_dict["eval_result"] else 0

            # add score
            if "overall" not in score_dict_model:
                score_dict_model["overall"] = []
            score_dict_model["overall"].append(judge_score)
            if judge_dict["benchmark_name"] not in score_dict_model:
                score_dict_model[judge_dict["benchmark_name"]] = []
            score_dict_model[judge_dict["benchmark_name"]].append(judge_score)

        for key, value in score_dict_model.items():
            score_dict_model[key] = round(sum(value) / len(value), 3)
        score_dict[model] = score_dict_model

        with open(
            os.path.join(args.model_response_dir, model, args.benchmark, args.version, "judge_results_mp_rule.jsonl"),
            "w",
        ) as f:
            for case in results:
                f.write(json.dumps(case) + "\n")

    if args.verbose:
        print(f"[Close-ended Multiple-choice Rule Parser]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")

    return score_dict


def compute_metric_closeended_multichoice(args):
    if args.multi_choice_parser == "model":
        return compute_metric_closeended_multichoice_modelparse(args)
    else:
        return compute_metric_closeended_multichoice_ruleparse(args)


def print_table(data_dict):
    # Create a table
    table = PrettyTable()

    # Set the column names
    table.field_names = ["Split", "Score"]

    # Add rows from the dictionary
    for key, value in data_dict.items():
        table.add_row([key, value])

    # Print the table
    print(table)


def compute_metric(args):
    score_dict_ff = compute_metric_closeended_freeform(args)
    score_dict_mp = compute_metric_closeended_multichoice(args)

    models_ff = set(score_dict_ff.keys())
    models_mp = set(score_dict_mp.keys())
    common_models = models_ff.intersection(models_mp)
    missing_models = models_ff.union(models_mp) - common_models
    if missing_models:
        print(
            f"Something went wrong when computing the free-form or multiple-choice "
            f"split of these models: \n{missing_models}\n\nA possible reason may be that they lack a model answer file. "
            "Skipping them..."
        )

    score_dict = {}
    for model in common_models:
        score_dir = os.path.join(
            args.model_response_dir,
            model,
            args.benchmark,
            args.version,
        )

        tmp_score_dict_model = {}
        for k in set(score_dict_mp[model].keys()).union(set(score_dict_ff[model].keys())):
            if k == "number_samples":
                continue
            sd_mp_l = score_dict_mp[model]["number_samples"].get(k, 0)
            sd_ff_l = score_dict_ff[model]["number_samples"].get(k, 0)

            tmp_score_dict_model[k] = score_dict_ff[model].get(k, 0) * (sd_ff_l / (sd_ff_l + sd_mp_l)) + score_dict_mp[
                model
            ].get(k, 0) * (sd_mp_l / (sd_ff_l + sd_mp_l))

        score_dict[model] = tmp_score_dict_model
        if args.multichoice_judge == args.freeform_judge:
            scorer = args.multichoice_judge
        else:
            scorer = f"mc{args.multichoice_judge}_ff{args.freeform_judge}"
        with open(os.path.join(score_dir, f"score_{scorer}.json"), "w") as f:
            f.write(json.dumps(tmp_score_dict_model, indent=4) + "\n")
        print_table(tmp_score_dict_model)

    target = f"score_{scorer}.json"
    print(f"Saving the model scores to {os.path.join(args.model_response_dir, target)} ...")
    with open(os.path.join(args.model_response_dir, target), "w") as f:
        f.write(json.dumps(score_dict, indent=4) + "\n")


def compute_metrics_p(args):
    # to be called in evaluate.py
    args.model_response_dir = args.output_dir
    args.models_to_eval = [args.model_name]
    compute_metric(args)


if __name__ == "__main__":
    set_seed()
    args = parse_args()
    compute_metric(args)
