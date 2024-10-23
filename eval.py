import argparse
import json
import logging
import os
import sys
import time
from typing import Optional, List, Dict, Type, Any
import random
import concurrent.futures
from abc import ABC, abstractmethod

import numpy as np
import torch

from lm_eval import utils
from lm_eval import evaluator as pretrain_evaluator
from lm_eval.tasks import TaskManager as PretrainTaskManager
from lm_eval.api.model import LM
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.utils import handle_non_serializable, simple_parse_args_string, sanitize_model_name
from lm_eval.__main__ import setup_parser, parse_eval_args
import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.api.task
import lm_eval.models
from eval.task import TaskManager as InstructTaskManager
from eval.eval_tracker import DCFTEvaluationTracker


def setup_custom_parser():
    """
    Create a custom argument parser that extends lm-eval-harness parser.
    """
    parser = setup_parser()
    db_group = parser.add_mutually_exclusive_group()
    
    db_group.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model UUID for direct database tracking"
    )
    
    db_group.add_argument(
        "--update_db_by_model_name",
        action="store_true",
        help="By default, databse is updated based on model uuid. Set this flag if you want to overwrite this and update database by searching for model name. Use model_name argument to specify the model name to update."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for direct database tracking. If not set, the model path will be used instead."
    )

    return parser


def evaluate(
    lm: LM,
    task_manager: InstructTaskManager,
    pretrain_task_manager: PretrainTaskManager,
    task_list: List[str],
    verbosity: str = "INFO",
    args=None,
    **eval_kwargs
) -> Dict[str, Dict]:
    """
    Evaluate the language model on the given tasks.

    Args:
        lm: The language model to evaluate
        task_manager: Task manager containing benchmark tasks
        pretrain_task_manager: Task manager for pretrain tasks
        task_list: List of task names to evaluate
        verbosity: Logging verbosity level
        args: Arguments for pretrain evaluation
        **eval_kwargs: Additional kwargs for evaluation

    Returns:
        Dictionary containing evaluation results for each task
    """
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    
    # Split tasks between benchmark and pretrain
    benchmark_tasks = [t for t in task_list if t in task_manager.tasks]
    pretrain_tasks = [t for t in task_list if t not in task_manager.tasks]

    
    if benchmark_tasks:
        eval_logger.info(f"Benchmark tasks to evaluate: {benchmark_tasks}")
    if pretrain_tasks:
        eval_logger.info(f"Pretrain tasks to evaluate: {pretrain_tasks}")
        
    results = {"results": {}}

    # Run benchmark evaluations with concurrent execution
    if benchmark_tasks:
        try:
            generate_methods = task_manager.get_list_generate_responses(benchmark_tasks)
            
            generation_results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_task = {
                    executor.submit(method, lm): task
                    for method, task in zip(generate_methods, benchmark_tasks)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        generation_results.append((task, result))
                    except Exception as e:
                        eval_logger.error(f"Error in generate_responses for {task}: {str(e)}")
                        generation_results.append((task, None))
            
            # Sort results back into original task order
            generation_results.sort(key=lambda x: benchmark_tasks.index(x[0]))
            generation_results = [r[1] for r in generation_results if r[1] is not None]
            
            # Get evaluate methods
            evaluate_methods = task_manager.get_list_evaluates(benchmark_tasks)
            
            # Run evaluation concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                evaluate_results = list(
                    executor.map(
                        lambda func_args: func_args[0](func_args[1]),
                        zip(evaluate_methods, generation_results)
                    )
                )
            
            # Store results
            for task, result in zip(benchmark_tasks, evaluate_results):
                results["results"][task] = result
                
        except Exception as e:
            eval_logger.error(f"Error in benchmark evaluation: {str(e)}")
    
    # Run pretrain evaluations if any exist
    if pretrain_tasks and args is not None:
        try:
            pretrain_results = pretrain_evaluator.simple_evaluate(
                model=args.model,
                model_args=args.model_args,
                tasks=pretrain_tasks,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                device=args.device,
                use_cache=args.use_cache,
                limit=args.limit,
                check_integrity=args.check_integrity,
                write_out=args.write_out,
                log_samples=args.log_samples,
                evaluation_tracker=args.evaluation_tracker if hasattr(args, 'evaluation_tracker') else None,
                system_instruction=args.system_instruction,
                apply_chat_template=args.apply_chat_template,
                fewshot_as_multiturn=args.fewshot_as_multiturn,
                gen_kwargs=args.gen_kwargs,
                task_manager=pretrain_task_manager,
                verbosity=args.verbosity,
                predict_only=args.predict_only,
                random_seed=args.seed[0] if hasattr(args, 'seed') else None,
                numpy_random_seed=args.seed[1] if hasattr(args, 'seed') else None,
                torch_random_seed=args.seed[2] if hasattr(args, 'seed') else None,
                fewshot_random_seed=args.seed[3] if hasattr(args, 'seed') else None,
            )
            results["results"].update(pretrain_results.get("results", {}))
        except Exception as e:
            eval_logger.error(f"Error in pretrain evaluation: {str(e)}")
            
    return results


def cli_evaluate(args: Optional[argparse.Namespace] = None) -> None:
    """Command-line interface for evaluating language models."""
    if not args:
        parser = setup_custom_parser()
        args = parse_eval_args(parser)


    # Initialize components
    task_list = args.tasks.split(",")
    task_manager = InstructTaskManager()
    pretrain_task_manager = PretrainTaskManager(args.verbosity, include_path=args.include_path)
    
    evaluation_tracker = setup_evaluation_tracker(args)

    utils.eval_logger.info(f"Selected Tasks: {[task for task in task_list]}")

    # Initialize model
    lm = initialize_model(args)   
    
    # Log experiment args
    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=args.model,
            model_args=args.model_args,
            system_instruction=args.system_instruction,
            chat_template=lm.chat_template(args.apply_chat_template),
            fewshot_as_multiturn=args.fewshot_as_multiturn,
        )

    # Initialize logging and environment
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Setup evaluation tracking
    if args.wandb_args:
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # Run evaluation
    results = evaluate(
        lm=lm,
        task_manager=task_manager,
        pretrain_task_manager=pretrain_task_manager,
        task_list=task_list,
        verbosity=args.verbosity,
        args=args,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        evaluation_tracker=evaluation_tracker,
    )
    
    # Add metadata and handle output
    add_results_metadata(results, args, lm)
    handle_evaluation_output(results, args, evaluation_tracker, wandb_logger if args.wandb_args else None)


def setup_evaluation_tracker(args: argparse.Namespace) -> DCFTEvaluationTracker:
    """Set up the evaluation tracker with proper arguments."""
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    return DCFTEvaluationTracker(args.output_path)


def initialize_model(args: argparse.Namespace) -> LM:
    """Initialize the language model based on arguments."""
    if isinstance(args.model, str):
        if args.model_args is None:
            args.model_args = ""
        
        lm = lm_eval.api.registry.get_model(args.model).create_from_arg_string(
            args.model_args,
            {
                "batch_size": args.batch_size,
                "max_batch_size": args.max_batch_size,
                "device": args.device,
            },
        )
    else:
        lm = args.model
        
    lm.model_identifier = sanitize_model_name(f"model_{args.model}_model_args_{args.model_args}")
    return lm


def add_results_metadata(results: Dict, args: argparse.Namespace, lm: LM) -> None:
    """Add metadata and configuration to results."""
    if lm.rank == 0:
        results["config"] = {
            "model": (
                args.model if isinstance(args.model, str)
                else args.model.config._name_or_path if hasattr(args.model, "config")
                else type(args.model).__name__
            ),
            "model_args": args.model_args,
            "batch_size": args.batch_size,
            "batch_sizes": (list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []),
            "device": args.device,
            "use_cache": args.use_cache,
            "limit": args.limit,
            # "bootstrap_iters": args.bootstrap_iters,
            "gen_kwargs": args.gen_kwargs,
            "random_seed": args.seed[0],
            "numpy_seed": args.seed[1],
            "torch_seed": args.seed[2],
            "fewshot_seed": args.seed[3],
        }
        
        if isinstance(lm, lm_eval.models.huggingface.HFLM):
            results["config"].update(lm.get_model_info())
            
        results["git_hash"] = get_git_commit_hash()
        results["date"] = time.time()
        add_env_info(results)
        add_tokenizer_info(results, lm)


def handle_evaluation_output(
    results: Dict,
    args: argparse.Namespace,
    evaluation_tracker: EvaluationTracker,
    wandb_logger: Optional[WandbLogger] = None,
) -> None:
    """Handle evaluation output, including logging and saving results."""
    if args.log_samples:
        samples = results.pop("samples")
        
    dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
    if args.show_config:
        print(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

    if wandb_logger:
        try:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            if args.log_samples:
                wandb_logger.log_eval_samples(samples)
        except Exception as e:
            eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

    evaluation_tracker.save_results_aggregated(
        results=results,
        samples=samples if args.log_samples else None
    )
    
    evaluation_tracker.update_evalresults_db(results, args.model_id, args.update_db_by_model_name, args.model_name)

    if args.log_samples:
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

    print(
        f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), "
        f"limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
        f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )

    if wandb_logger:
        wandb_logger.run.finish()



if __name__ == "__main__":
    cli_evaluate()
