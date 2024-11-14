import argparse
import json
import logging
import os
import sys
import time
from typing import Optional, List, Dict 

import concurrent.futures
import torch.distributed as dist

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

    db_group.add_argument("--model_id", type=str, default=None, help="Model UUID for direct database tracking")

    parser.add_argument("--use-database", action="store_true", help="Where to use DCFT Database to track results.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for direct database tracking. If not set, the model path will be used instead.",
    )

    db_group.add_argument(
        "--is_external_model",
        action="store_true",
        help="By default, the model is stored as internal in the database. If set, this is overwritten to external.",
    )

    parser.add_argument(
        "--creation_location",
        type=str,
        default="NA",
        help="Specifies which compute server is used for evaluating the model.",
    )

    parser.add_argument(
        "--created_by",
        type=str,
        default="NA",
        help="Specifies who evaluates the model.",
    )

    parser.add_argument(
        "--annotator_model",
        type=str,
        default="auto",
        help="Judge model used to evaluate generations. Example: gpt-4o-mini-2024-07-18",
    )
    return parser


def evaluate(
    lm: LM,
    task_manager: InstructTaskManager,
    pretrain_task_manager: PretrainTaskManager,
    task_list: List[str],
    verbosity: str = "INFO",
    args=None,
    **eval_kwargs,
) -> Dict[str, Dict]:
    """
    Evaluate the language model on the given tasks.
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

    starting_time = time.time()
    # Run benchmark evaluations - sequential generation, parallel evaluation
    if benchmark_tasks:
        # Sequential generation since it's GPU-bound
        generate_methods = task_manager.get_list_generate_responses(benchmark_tasks)
        generation_results = []
        valid_tasks = []  # Keep track of valid tasks
        for method, task in zip(generate_methods, benchmark_tasks):
            result = method(lm)
            if result is not None:  # Only keep valid results and their corresponding tasks
                generation_results.append(result)
                valid_tasks.append(task)
        # Get evaluation methods only for valid tasks
        evaluate_methods = task_manager.get_list_evaluates(valid_tasks)
        cpu_count = os.cpu_count()

        max_workers = min(len(valid_tasks), cpu_count * 2)
        if lm.world_size <= 1 or lm.rank == 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                evaluate_results = list(
                    executor.map(
                        lambda func_args: func_args[0](func_args[1]), zip(evaluate_methods, generation_results)
                    )
                )

            # Store results using valid tasks for correct mapping
            for task, result in zip(valid_tasks, evaluate_results):
                results["results"][task] = result
    ending_time = time.time()
    results['Total Time Taken'] = ending_time - starting_time

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
                evaluation_tracker=args.evaluation_tracker if hasattr(args, "evaluation_tracker") else None,
                system_instruction=args.system_instruction,
                apply_chat_template=args.apply_chat_template,
                fewshot_as_multiturn=args.fewshot_as_multiturn,
                gen_kwargs=args.gen_kwargs,
                task_manager=pretrain_task_manager,
                verbosity=args.verbosity,
                predict_only=args.predict_only,
                random_seed=args.seed[0] if hasattr(args, "seed") else None,
                numpy_random_seed=args.seed[1] if hasattr(args, "seed") else None,
                torch_random_seed=args.seed[2] if hasattr(args, "seed") else None,
                fewshot_random_seed=args.seed[3] if hasattr(args, "seed") else None,
            )
            if pretrain_results is not None:
                results["results"].update(pretrain_results.get("results", {}))
        except Exception as e:
            eval_logger.error(f"Error in pretrain evaluation: {str(e)}")

    return results


def update_model_args_with_name(model_args: str, model_name: str) -> str:
    """
    Update model_args string to include pretrained model name if not already present.

    Args:
        model_args: Original model args string
        model_name: Model name to add

    Returns:
        str: Updated model args string
    """
    if not model_args:
        return f"pretrained={model_name}"

    args_dict = simple_parse_args_string(model_args)
    if "pretrained" not in args_dict:
        return f"pretrained={model_name},{model_args}"
    return model_args


def cli_evaluate(args: Optional[argparse.Namespace] = None) -> None:
    """
    Command-line interface for evaluating language models.

    Args:
        args: Command line arguments. If None, will parse from sys.argv
    """
    # Parse arguments if not provided
    if not args:
        parser = setup_custom_parser()
        args = parse_eval_args(parser)

    # Initialize evaluation tracker
    evaluation_tracker = setup_evaluation_tracker(args)

    # If model_id is provided, lookup model name from database
    if args.model_id:
        if not args.use_database:
            raise ValueError("--use-database must be set to use --model-id.")
        try:
            model_name = evaluation_tracker.get_model_name_from_db(args.model_id)
            args.model_args = update_model_args_with_name(args.model_args or "", model_name)
            utils.eval_logger.info(f"Retrieved model name from database: {model_name}")
        except Exception as e:
            utils.eval_logger.error(f"Failed to retrieve model name from database: {str(e)}")
            sys.exit(1)
    elif args.model_name:
        model_name = args.model_name
        args.model_args = update_model_args_with_name(args.model_args or "", model_name)

    # Initialize tasks
    task_list = args.tasks.split(",")
    task_manager = InstructTaskManager(annotator_model=args.annotator_model)
    pretrain_task_manager = PretrainTaskManager(args.verbosity, include_path=args.include_path)

    utils.eval_logger.info(f"Selected Tasks: {[task for task in task_list]}")

    # Initialize model
    try:
        lm = initialize_model(args)
    except Exception as e:
        utils.eval_logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)

    # Log experiment configuration
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

    # Setup wandb logging if requested
    wandb_logger = None
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
        annotator_model=args.annotator_model,
        evaluation_tracker=evaluation_tracker,
    )

    # Add metadata to results
    if lm.rank == 0:
        add_results_metadata(results, args, lm)
        handle_evaluation_output(results, args, evaluation_tracker, wandb_logger)

    if dist.is_initialized():
        dist.destroy_process_group()


def setup_evaluation_tracker(args: argparse.Namespace) -> DCFTEvaluationTracker:
    """Set up the evaluation tracker with proper arguments."""
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    return DCFTEvaluationTracker(args.output_path, args.use_database)


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
    results["config"] = {
        "model": (
            args.model
            if isinstance(args.model, str)
            else args.model.config._name_or_path if hasattr(args.model, "config") else type(args.model).__name__
        ),
        "model_args": args.model_args,
        "batch_size": args.batch_size,
        "batch_sizes": (list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []),
        "device": args.device,
        "use_cache": args.use_cache,
        "limit": args.limit,
        "annotator_model": args.annotator_model,
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
            utils.eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

    evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None)
    if args.use_database:
        evaluation_tracker.update_evalresults_db(
            results,
            args.model_id,
            args.model_name,
            args.creation_location,
            args.created_by,
            args.is_external_model,
        )

    if args.log_samples:
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

    utils.eval_logger.info(
        f"Eval arugments: {args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), "
        f"limit: {args.limit}, num_fewshot: {args.num_fewshot}, annotator_model: {args.annotator_model}, "
        f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )

    if wandb_logger:
        wandb_logger.run.finish()


if __name__ == "__main__":
    cli_evaluate()
