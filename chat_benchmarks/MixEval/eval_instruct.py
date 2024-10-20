from typing import Dict, List, Any
import mix_eval
from lm_eval.api.model import LM
import torch
from tqdm import tqdm
import os
import json

from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from argparse import Namespace
from mix_eval.evaluate import parse_args
from mix_eval.utils.dataset import get_eval_dataset
from mix_eval.compute_metrics import compute_metrics_p
from mix_eval.utils.common_utils import cache_status, read_status, dict_equal

os.environ["MODEL_PARSER_API"] = os.getenv("OPENAI_API_KEY")

# TODO: Replace this with DB access
os.makedirs("eval/chat_benchmarks/MixEval/results/", exist_ok=True)


# TODO: Replace these args with eval args manager
def get_default_args():
    """
    Retrieve the default arguments for evaluation.

    Returns:
        Dict[str, Any]: A dictionary of default argument values.
    """
    parser = parse_args(return_parser=True)
    return {action.dest: action.default for action in parser._actions if action.dest != "help"}


args = get_default_args()
args.update(
    {
        "output_dir": "eval/chat_benchmarks/MixEval/results/",
        "benchmark": "mixeval",
        "version": "2024-06-01",
        "batch_size": 8,
        "max_gpu_memory": "60GB",
        "data_path": "eval/chat_benchmarks/MixEval/mix_eval/data/",
        "output_dir": "eval/chat_benchmarks/MixEval/results/",
        "api_parallel_num": 32,
        # Original judges in MixEval:
        "multichoice_judge": "gpt-3.5-turbo-0125",
        "freeform_judge": "gpt-3.5-turbo-0125",
        # New judges:
        # "multichoice_judge": "gpt-4o-mini",
        # "freeform_judge": "gpt-4o-mini",
        "verbose": False,
    }
)
args = Namespace(**args)
# end TODO


def eval_instruct(model: LM) -> Dict[str, Any]:
    """
    Evaluate the given model on predefined splits.

    Args:
        model (LM): A dummy model from lm_eval, not used but required to fit lm_eval API

    Returns:
        Dict[str, Any]: A dictionary containing evaluation results for each split.
    """
    splits = ["close_freeform", "close_multichoice"]
    out_dict = {}
    for split in splits:
        out_dict[split] = eval_instruct_split(model, split)
    return out_dict


def eval_instruct_split(model: LM, split: str) -> Dict[str, Any]:
    """
    Evaluate the model on a specific data split.

    Args:
        model (LM): The language model to evaluate.
        split (str): The data split to evaluate on.

    Returns:
        Dict[str, Any]: The evaluation results for the specified split.
    """
    time_elapsed = 0
    start_time = time.time()
    args.split = split
    if "model_identifier" in model.__dict__:
        model_name = (
            model.model_identifier.split("=")[-1]
            .split("__")[-1]
            .replace("Meta-", "")
            .replace("-", "_")
            .lower()
            .replace(".", "")
        )
        args.model_name = model_name
    else:
        args.model_name = model.model.__class__.__name__

    # TODO: File saving (and verification that eval already exists) should be replaced with DB access
    response_file = os.path.join(
        args.output_dir, args.model_name, args.benchmark, args.version, f"{args.model_name}_{args.split}.jsonl"
    )
    os.makedirs(os.path.dirname(response_file), exist_ok=True)

    # if the response file exists, check if it can resume from last run
    resume = False
    if os.path.exists(response_file):
        status = read_status(args)
        not_gen_args = [
            "freeform_judge",
            "multichoice_judge",
            "max_gpu_memory",
            "api_parallel_num",
            "max_tasks",
            "batch_size",
            "api_parallel_num",
        ]
        subdict_status = {k: v for k, v in status["args"].items() if not k in not_gen_args}
        subdict_args = {k: v for k, v in args.__dict__.items() if not k in not_gen_args}
        if not dict_equal(subdict_status, subdict_args):
            raise ValueError(
                f"The model response file {response_file} already exists. The cached arguments are "
                "different from those in the current run. Please check."
            )
        if status["status"]["status"] == "complete":
            print(f"The evaluation for {args.model_name}'s {args.split} " "split is already complete. Skipping.")
            with open(response_file) as f:
                output = [json.loads(line) for line in f]
            return output
        with open(response_file) as f:
            lines = f.readlines()
            if len(lines) == (status["status"]["batch_id"] + 1) * args.batch_size:
                resume = True
                time_elapsed += time.time() - start_time + status["status"]["time_elapsed"]
                start_time = time.time()
                print(f"Resuming from last run: \n{status}")
            else:
                raise ValueError(
                    f"The response file [{response_file}] has different "
                    "lines as recorded in cached metadadta. Please check the response file. "
                    "You might consider delete the response and metadata file to start from scratch."
                )

    model = mix_eval.api.registry.get_model(args.model_name)(args)
    eval_dataset = get_eval_dataset(args)

    dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, collate_fn=lambda x: x
    )

    with torch.no_grad():
        for b_id, batch in enumerate(tqdm(dataloader, desc="Evaluating batches", unit="batch")):
            if resume:
                if status["status"]["batch_id"] >= b_id:
                    continue
                else:
                    resume = False

            if args.verbose:
                _start_time = time.time()
            model.get_responses(batch, response_file)
            if args.verbose:
                _finish_time = time.time()
                print(f"Batch {b_id} finished in {_finish_time - _start_time} seconds.")

            time_elapsed += time.time() - start_time
            start_time = time.time()

            status = {"batch_id": b_id, "time_elapsed": time_elapsed, "status": "in progress"}
            cache_status(args, status)

    status = {"batch_id": b_id, "time_elapsed": time_elapsed, "status": "complete"}
    cache_status(args, status)
    print(
        f"Finished evaluating {args.model_name}'s {args.split} split. " f"Used {round(time_elapsed / 60, 2)} minutes."
    )

    result_path = os.path.join(
        args.output_dir, args.model_name, args.benchmark, args.version, f"{args.model_name}_{args.split}.jsonl"
    )

    with open(result_path, "r") as f:
        results = [json.loads(line) for line in f]

    return results


def evaluate(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate the model outputs and compute metrics.

    Args:
        results (Dict[str, Any]): The model outputs to evaluate.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics for each split.
    """

    # Compute metrics
    compute_metrics_p(args)

    score_dir = os.path.join(
        args.model_response_dir,
        args.model_name,
        args.benchmark,
        args.version,
    )
    # find score file
    result_files = [f for f in os.listdir(score_dir) if f.startswith("score") and f.endswith(".json")]
    if "score_gpt-4o-mini.json" in result_files:
        score_file = "score_gpt-4o-mini.json"
        judge_model = "gpt-4o-mini"
    elif "score.json" in result_files:
        score_file = "score.json"
        judge_model = (
            args.multichoice_judge
            if args.mulitchoice_judge == args.freeform_judge
            else f"mc{args.multichoice_judge}_ff{args.freeform_judge}"
        )
    else:
        raise ValueError(f"Expected 'score_gpt-4o-mini.json' or 'score.json' in {score_dir}, but found {result_files}")

    with open(os.path.join(score_dir, score_file), "r") as f:
        metrics = json.load(f)

    for result_file in result_files:
        if result_file.startswith("judge_results_ff"):
            with open(os.path.join(score_dir, result_file), "r") as f:
                results_ff = json.load(f)
        elif result_file.startswith("judge_results_mp"):
            with open(os.path.join(score_dir, result_file), "r") as f:
                results_mp = json.load(f)
    return {
        judge_model: {
            "metrics": metrics,
            "judge_answers": {"close_freeform": results_ff, "close_multichoice": results_mp},
        },
        "samples": results,
    }
