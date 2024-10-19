import argparse
import json
import logging
import os
import sys
import time
from typing import Optional, List, Dict
import random
import concurrent.futures

import numpy as np
import torch

from eval.task import TaskManager as InstructTaskManager
from eval.eval_tracker import DCFTEvaluationTracker
from lm_eval import utils
from lm_eval import evaluator as pretrain_evaluator
from lm_eval.tasks import TaskManager as PretrainTaskManager
from lm_eval.api.model import LM
from lm_eval.loggers import WandbLogger
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string, sanitize_model_name
from lm_eval.__main__ import setup_parser, parse_eval_args
import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.api.task
import lm_eval.models
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash


def evaluate(
    lm: LM,
    task_manager: InstructTaskManager,
    task_list: List[str],
    verbosity: str = "INFO",
) -> Dict[str, Dict]:
    """
    Evaluate the language model on the given tasks.
    Args:
        lm (LM): The language model to evaluate.
        task_manager (InstructTaskManager): The task manager containing evaluation instructions.
        task_list (List[str]): List of task names to evaluate.
        verbosity (str, optional): Logging verbosity level. Defaults to "INFO".
    Returns:
        Dict[str, Dict]: A dictionary containing evaluation results for each task.
    """
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        eval_instruct_results = list(
            executor.map(lambda task: task(lm), task_manager.get_list_eval_instructs(task_list))
        )

        evaluate_results = list(
            executor.map(
                lambda func_args: func_args[0](func_args[1]),
                zip(task_manager.get_list_evaluates(task_list), eval_instruct_results),
            )
        )

    return {task: result for task, result in zip(task_list, evaluate_results)}


def setup_random_seeds(random_seed, numpy_random_seed, torch_random_seed):
    seed_messages = []
    if random_seed is not None:
        seed_messages.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)
    if numpy_random_seed is not None:
        seed_messages.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)
    if torch_random_seed is not None:
        seed_messages.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)
    return " | ".join(seed_messages)


def initialize_model(model, model_args, batch_size, max_batch_size, device):
    if isinstance(model, str):
        model_args = model_args or ""
        model_args_dict = simple_parse_args_string(model_args)
        utils.eval_logger.info(f"Initializing {model} model, with arguments: {model_args_dict}")
        return lm_eval.api.registry.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    elif not isinstance(model, lm_eval.api.model.LM):
        raise TypeError(
            f"The value of `model` passed to simple_evaluate() was of type {type(model)}, "
            "but is required to be a subclass of lm_eval.api.model.LM"
        )
    utils.eval_logger.info("Using pre-initialized model")
    return model


def cli_evaluate(args: Optional[argparse.Namespace] = None) -> None:
    if not args:
        parser = setup_parser()
        args = parse_eval_args(parser)

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Validate arguments
    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError("When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set.")

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()

    # Initialize components
    task_list = args.tasks.split(",")
    task_manager = InstructTaskManager()
    evaluation_tracker = DCFTEvaluationTracker(args.output_path)

    # Model setup
    lm = initialize_model(args.model, args.model_args, args.batch_size, args.max_batch_size, args.device)
    lm.model_identifier = sanitize_model_name(f"model_{args.model}_model_args_{args.model_args}")

    pretrain_task_manager = PretrainTaskManager(args.verbosity, include_path=args.include_path)

    instruct_task_names = [task for task in task_list if task in task_manager.tasks]
    pretrain_task_names = [task for task in task_list if task not in task_manager.tasks]

    eval_logger.info(f"Selected Tasks: {[task for task in task_list]}")
    start_date = time.time()
    if len(pretrain_task_names) > 0:
        pretrain_results = pretrain_evaluator.simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=pretrain_task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            use_cache=args.use_cache,
            limit=args.limit,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            log_samples=args.log_samples,
            evaluation_tracker=evaluation_tracker,
            system_instruction=args.system_instruction,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            gen_kwargs=args.gen_kwargs,
            task_manager=pretrain_task_manager,
            verbosity=args.verbosity,
            predict_only=args.predict_only,
            random_seed=args.seed[0],
            numpy_random_seed=args.seed[1],
            torch_random_seed=args.seed[2],
            fewshot_random_seed=args.seed[3],
        )

    # Log experiment args
    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=args.model,
            model_args=args.model_args,
            system_instruction=args.system_instruction,
            chat_template=lm.chat_template(args.apply_chat_template),
            fewshot_as_multiturn=args.fewshot_as_multiturn,
        )
    results = {}
    if len(instruct_task_names) > 0:
        results["results"] = evaluate(lm, task_manager=task_manager, task_list=instruct_task_names, verbosity=args.verbosity)

    # Setup random seeds
    seed_message = setup_random_seeds(args.seed[0], args.seed[1], args.seed[2])
    if seed_message:
        eval_logger.info(seed_message)
    
    # Process results
    if lm.rank == 0:
        results = process_results(results, lm, args, start_date)

    if results is not None or pretrain_results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        if pretrain_results is not None and results is not None:
            results["results"].update(pretrain_results["results"])
        elif pretrain_results is not None and results is None:
            results = pretrain_results

        dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
        if args.show_config:
            print(dumped)

        # Log results
        log_results(results, args, evaluation_tracker, samples if args.log_samples else None)


def process_results(results, lm, args, start_date):
    config = {
        "model": args.model if isinstance(args.model, str) else type(args.model).__name__,
        "model_args": args.model_args,
        "batch_size": args.batch_size,
        "batch_sizes": (list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []),
        "device": args.device,
        "use_cache": args.use_cache,
        "limit": args.limit,
        "gen_kwargs": args.gen_kwargs,
        "random_seed": args.seed[0],
        "numpy_seed": args.seed[1],
        "torch_seed": args.seed[2],
        "fewshot_seed": args.seed[3],
    }

    if isinstance(lm, lm_eval.models.huggingface.HFLM):
        config.update(lm.get_model_info())

    results = {
        "results": results,
        "config": config,
        "git_hash": get_git_commit_hash(),
        "date": start_date,
    }
    add_env_info(results)
    add_tokenizer_info(results, lm)

    return results


def log_results(results, args, evaluation_tracker, samples=None):
    if args.show_config:
        print(json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False))

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

    if args.wandb_args:
        try:
            wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            if samples:
                wandb_logger.log_eval_samples(samples)
        except Exception as e:
            utils.eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

    evaluation_tracker.save_results_aggregated(results=results, samples=samples)
    evaluation_tracker.update_evalresults_db(results)

    if samples:
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

    print(
        f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )


if __name__ == "__main__":
    cli_evaluate()
