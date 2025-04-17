#!/usr/bin/env python
import argparse
import hashlib
import os
import re
import socket
import subprocess
import time

from huggingface_hub import HfApi

clusters = [
    {
        "name": "capella",
        "hostname_pattern": r"c\d",
        "eval_sbatch_filename": "simple_zih.sbatch",
        "gpus_per_node": 4,
    },
    {
        "name": "vista",
        "hostname_pattern": r".*?.vista.tacc.utexas.edu",
        "eval_sbatch_filename": "simple_tacc.sbatch",
        "gpus_per_node": 1,
    },
]


def detect_hpc() -> dict:
    """Automatically detect the HPC based on hostname"""
    hostname = socket.gethostname()
    for cluster in clusters:
        if re.compile(cluster["hostname_pattern"]).match(hostname):
            print(f"Automatically detected HPC: {cluster['name']}")
            return cluster
    raise ValueError(f"HPC not recognized for hostname {hostname}")


def generate_evaluation_dataset_hash(tasks, system_instruction=None):
    """Generate a 4-character hash from the task list and system instruction."""
    print(f"Tasks to evaluate: {', '.join(tasks)}")
    tasks_str = ",".join(sorted(tasks))
    hash_obj = hashlib.md5((tasks_str + (system_instruction or "")).encode())
    return hash_obj.hexdigest()[:4]


def check_dataset_exists(repo_id):
    """Check if a dataset repository exists on Hugging Face using the HfApi."""
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        return True
    except Exception:
        return False


def execute_command(cmd):
    """Execute a shell command and return the output."""
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy(), universal_newlines=True
    )  # noqa
    stdout, stderr = process.communicate()
    return_code = process.returncode
    if return_code != 0:
        raise Exception(f"Command failed with return code {return_code} and error: {stderr.strip()}")
    return stdout.strip()


def create_evaluation_dataset(tasks, eval_dataset_hash, system_instruction=None):
    """Create or use cached evaluation dataset."""
    cached_dataset_id = f"mlfoundations-dev/evalset_{eval_dataset_hash}"
    if check_dataset_exists(cached_dataset_id):
        print(f"Using cached evaluation dataset: {cached_dataset_id}")
    else:
        print("Creating new evaluation dataset...")
        tasks_str = ",".join(tasks)
        cmd = f"python -m eval.eval --model upload_to_hf --tasks {tasks_str} --model_args repo_id={cached_dataset_id} --output_path logs"
        if system_instruction:
            cmd += f" --system_instruction '{system_instruction}'"
        execute_command(cmd)
        print(f"Evaluation dataset created: https://huggingface.co/datasets/{cached_dataset_id}")
    return cached_dataset_id


def launch_sbatch(sbatch_content, new_sbatch_file):
    # Write the sbatch file
    with open(new_sbatch_file, "w") as f:
        f.write(sbatch_content)
    print(f"Created sbatch file: {new_sbatch_file}")

    # Launch the sbatch job
    stdout = execute_command(f"sbatch {new_sbatch_file}")
    job_id_match = re.search(r"Submitted batch job (\d+)", stdout)
    if job_id_match:
        job_id = job_id_match.group(1)
    else:
        raise Exception("Could not determine job ID from sbatch output.")
    return job_id


def main():
    parser = argparse.ArgumentParser(description="Distributed Evaluation Job Manager")
    parser.add_argument(
        "--tasks",
        type=str,
        default="AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name/path to evaluate")
    parser.add_argument("--num_shards", type=int, default=16, help="Number of shards for distributed evaluation")
    parser.add_argument(
        "--max-job-duration",
        type=int,
        default=4,
        help="Maximum job duration in hours",
    )
    parser.add_argument("--system_instruction", type=str, default=None, help="System instruction for the model")
    parser.add_argument("--timestamp", action="store_true", help="Add a timestamp to the output evaluation dataset")
    args = parser.parse_args()

    # Generate evaluation dataset hash
    tasks = [task.strip() for task in args.tasks.split(",")]
    evaluation_dataset_hash = generate_evaluation_dataset_hash(tasks, args.system_instruction)

    # Download or create input dataset
    input_dataset = create_evaluation_dataset(tasks, evaluation_dataset_hash, args.system_instruction)

    # Create output dataset name
    if args.timestamp:
        timestamp = str(int(time.time()))
        suffix = f"_{timestamp}_eval_{evaluation_dataset_hash}"
    else:
        suffix = f"_eval_{evaluation_dataset_hash}"
    output_dataset_name = args.model_name.split("/")[-1] + suffix
    output_dataset = f"mlfoundations-dev/{output_dataset_name}"
    print(f"Output dataset: {output_dataset}")

    # Create output log dir
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Determine sbatch filename based on HPC
    cluster = detect_hpc()
    eval_sbatch_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cluster["eval_sbatch_filename"])

    # Determine number of nodes
    if args.num_shards % cluster["gpus_per_node"] != 0:
        raise ValueError(
            f"Number of shards ({args.num_shards}) must be a multiple of the number of GPUs per node ({cluster['gpus_per_node']})"
        )
    num_nodes = int(args.num_shards / cluster["gpus_per_node"])

    # Create sbatch
    args_dict = vars(args)
    args_dict["num_nodes"] = num_nodes
    args_dict["time_limit"] = f"{args.max_job_duration:02d}:00:00"
    args_dict["job_name"] = f"{output_dataset_name}"
    args_dict["input_dataset"] = input_dataset
    args_dict["output_dataset"] = output_dataset
    args_dict["logs_dir"] = logs_dir
    args_dict["output_dataset_name"] = output_dataset_name
    args_dict["tasks_str"] = args.tasks
    with open(eval_sbatch_path, "r") as f:
        sbatch_content = f.read()
    curly_brace_pattern = r"(?<!\$)\{([^{}]*)\}"
    sbatch_content = re.sub(curly_brace_pattern, lambda m: str(args_dict[m.group(1)]), sbatch_content)

    # Launch sbatch
    new_sbatch_file = os.path.join(logs_dir, f"{output_dataset_name}.sbatch")
    job_id = launch_sbatch(sbatch_content, new_sbatch_file)
    print(f"Launched sbatch job with ID: {job_id}")
    print(f"Logs: {args_dict['logs_dir']}/{args_dict['job_name']}_{job_id}*.out")


if __name__ == "__main__":
    main()
