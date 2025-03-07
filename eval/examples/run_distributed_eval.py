#!/usr/bin/env python

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import colorama
from colorama import Fore, Style
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Initialize colorama
colorama.init()


def print_colored(text, color=Fore.WHITE, style=Style.NORMAL, end="\n"):
    """Print text with color and style."""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)


def print_header(text):
    """Print a header with nice formatting."""
    print_colored(f"\n{'-' * 80}", Fore.CYAN, Style.BRIGHT)
    print_colored(f" {text}", Fore.CYAN, Style.BRIGHT)
    print_colored(f"{'-' * 80}", Fore.CYAN, Style.BRIGHT)


def print_success(text):
    """Print a success message."""
    print_colored(f"✓ {text}", Fore.GREEN)


def print_warning(text):
    """Print a warning message."""
    print_colored(f"⚠ {text}", Fore.YELLOW)


def print_error(text):
    """Print an error message."""
    print_colored(f"✗ {text}", Fore.RED, Style.BRIGHT)


def print_info(text):
    """Print an info message."""
    print_colored(f"ℹ {text}", Fore.BLUE)


def execute_command(cmd, env=None, verbose=True):
    """Execute a shell command and return the output."""
    if verbose:
        print_info(f"Running: {cmd}")

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True
    )

    stdout, stderr = process.communicate()
    return_code = process.returncode

    if return_code != 0 and verbose:
        print_error(f"Command failed with return code {return_code}")
        print_error(f"Error: {stderr.strip()}")

    return stdout.strip(), stderr.strip(), return_code


def check_required_env_vars():
    """Check if required environment variables are set."""
    print_header("Checking Environment Variables")

    required_vars = ["HF_TOKEN", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER"]
    missing_vars = []

    for var in required_vars:
        if os.environ.get(var) is None:
            missing_vars.append(var)

    if missing_vars:
        print_error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print_info("Please set these variables in your .env file and try again.")
        return False

    print_success("All required environment variables are set.")
    return True


def activate_conda_env():
    """Activate the conda environment."""
    print_header("Activating Conda Environment")

    # Check if we're already in the evalchemy conda environment
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env == "evalchemy":
        print_success("Already in evalchemy conda environment.")
        return True

    # Try to activate the evalchemy conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX_1", os.environ.get("CONDA_PREFIX"))
    if not conda_prefix:
        print_error("Conda environment not detected.")
        return False

    # Since we can't directly activate conda environment in a script,
    # we'll check if the environment exists
    cmd = "conda env list | grep evalchemy"
    stdout, _, return_code = execute_command(cmd)

    if return_code != 0 or "evalchemy" not in stdout:
        print_error("evalchemy conda environment not found.")
        print_info("Please create and activate the evalchemy conda environment manually.")
        return False

    print_warning("Script is running in conda environment: " + current_env)
    print_info("This script cannot automatically activate a different conda environment.")
    print_info("Please run this script from the evalchemy conda environment.")
    return True


def generate_task_hash(tasks):
    """Generate a 4-character hash from the task list."""
    tasks_str = ",".join(sorted(tasks))
    hash_obj = hashlib.md5(tasks_str.encode())
    return hash_obj.hexdigest()[:4]


def check_dataset_exists(repo_id):
    """Check if a dataset repository exists on Hugging Face using the HfApi."""
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        return True
    except Exception:
        return False


def create_evaluation_dataset(tasks):
    """Create or use cached evaluation dataset."""
    print_header("Preparing Evaluation Dataset")

    # Generate a cached dataset name based on tasks
    task_hash = generate_task_hash(tasks)
    cached_dataset_id = f"mlfoundations-dev/evalset_{task_hash}"

    # Check if the cached dataset exists
    if check_dataset_exists(cached_dataset_id):
        print_success(f"Using cached evaluation dataset: {cached_dataset_id}")
        return cached_dataset_id

    # If not, create a new evaluation dataset
    print_info("Creating new evaluation dataset...")
    print_warning(
        "This may take a while the first time the eval datasets are downloaded and parsed. Consider running locally with more cpus."
    )
    tasks_str = ",".join(tasks)
    cmd = f"OPENAI_API_KEY=NONE python -m eval.eval --model upload_to_hf --tasks {tasks_str} --model_args repo_id={cached_dataset_id} --output_path logs"

    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_error("Failed to create evaluation dataset.")
        print_error(f"Error: {stderr}")
        return False

    print_success(f"Evaluation dataset created at: {cached_dataset_id}")
    return cached_dataset_id


def prepare_for_sbatch(input_repo_id, output_repo_id, model_name):
    """Prepare for the sbatch job."""
    print_header("Preparing for SBATCH Job")

    # Extract the repository name without the organization
    repo_name = output_repo_id.split("/")[-1]

    # Create a logs directory
    logs_dir = os.path.join("logs", repo_name)
    os.makedirs(logs_dir, exist_ok=True)
    print_success(f"Created logs directory: {logs_dir}")

    # Set HF_HUB environment variable
    hf_hub = os.environ.get("HF_HOME", "/data/horse/ws/ryma833h-DCFT_Shared/hub")
    os.environ["HF_HUB"] = hf_hub
    print_success(f"Set HF_HUB to: {hf_hub}")

    # Download the dataset
    print_info(f"Downloading dataset from: {input_repo_id}")
    cmd = f"huggingface-cli download {input_repo_id} --repo-type dataset"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_warning(f"Dataset download may have issues: {stderr}")
    else:
        print_success("Dataset downloaded successfully.")

    # Download the target model
    # print_info(f"Ensuring model is available: {model_name}")
    # cmd = f"python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{model_name}')\""
    # stdout, stderr, return_code = execute_command(cmd)
    print_info(f"Downloading model from: {model_name}")
    cmd = f"huggingface-cli download {model_name}"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_warning(f"Model download may have issues: {stderr}")
    else:
        print_success("Model download check successful.")

    return logs_dir


def launch_sbatch(model_name, input_dataset, output_dataset, num_shards, logs_dir, tasks_str):
    """Launch the sbatch job."""
    print_header("Launching SBATCH Job")

    # Create a temporary sbatch script with the correct parameters
    temp_sbatch_file = os.path.join(logs_dir, "job.sbatch")

    with open("eval/examples/zih_sharded_ALL_job_array.sbatch", "r") as f:
        sbatch_content = f.read()

    # Replace parameters in the sbatch script
    sbatch_content = sbatch_content.replace("#SBATCH --array=0-127", f"#SBATCH --array=0-{num_shards-1}")
    sbatch_content = sbatch_content.replace('export TASK="REASONING"', f'export TASK="{tasks_str}"')
    sbatch_content = sbatch_content.replace(
        'export INPUT_DATASET="mlfoundations-dev/${TASK}_evalchemy"', f'export INPUT_DATASET="{input_dataset}"'
    )
    sbatch_content = sbatch_content.replace(
        'export OUTPUT_DATASET="mlfoundations-dev/${TASK}_evalchemy_${GLOBAL_SIZE}shards_${MODEL_NAME_SHORT}"',
        f'export OUTPUT_DATASET="{output_dataset}"',
    )
    sbatch_content = sbatch_content.replace(
        'export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"', f'export MODEL_NAME="{model_name}"'
    )

    # Add output log path
    sbatch_content = sbatch_content.replace(
        "#SBATCH --mem=64G", f"#SBATCH --mem=64G\n#SBATCH --output={logs_dir}/%A_%a.out"
    )

    with open(temp_sbatch_file, "w") as f:
        f.write(sbatch_content)

    print_success(f"Created temporary sbatch file: {temp_sbatch_file}")

    # Launch the sbatch job
    cmd = f"sbatch {temp_sbatch_file}"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_error(f"Failed to launch sbatch job: {stderr}")
        return None

    # Extract the job ID from the output
    job_id_match = re.search(r"Submitted batch job (\d+)", stdout)
    if job_id_match:
        job_id = job_id_match.group(1)
        print_success(f"SBATCH job submitted with ID: {job_id}")
        return job_id
    else:
        print_error("Could not determine job ID from sbatch output.")
        return None


def monitor_job(job_id, logs_dir, num_shards, watchdog_interval_min=1):
    """Monitor the slurm job and show progress."""
    print_header("Monitoring Job Progress")

    # Determine the log file pattern based on the job ID
    log_pattern = f"{logs_dir}/{job_id}_*.out"

    # Define job states
    running_states = ["RUNNING", "PENDING", "REQUEUED", "CONFIGURING", "COMPLETING", "RESIZING", "SUSPENDED", "REVOKED"]
    failed_states = [
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "DEADLINE",
        "BOOT_FAIL",
        "SPECIAL_EXIT",
    ]

    time.sleep(5)
    counter = 0
    try:
        while True:
            # Get job state counts
            cmd = f"sacct -j {job_id} -X --format=State --noheader"
            stdout, _, _ = execute_command(cmd, verbose=False)

            if not stdout.strip():
                print_warning("No job state information available yet. Waiting...")
                time.sleep(10)
                continue

            states = stdout.strip().split("\n")

            # Count states by category
            running_count = sum(1 for state in states if any(s in state for s in running_states))
            completed_count = sum(1 for state in states if "COMPLETED" in state)
            failed_count = sum(1 for state in states if any(s in state for s in failed_states))
            total_count = len(states)

            # Show job summary
            print_info(
                f"({counter*watchdog_interval_min}m) Job Status: {completed_count} completed, {running_count} running, {failed_count} failed, {total_count} total"
            )

            # Show failed jobs if any
            if failed_count > 0:
                # Create regex pattern for failed states
                failed_pattern = "|".join(failed_states)
                cmd = f"sacct -j {job_id} -X --format=JobID%20,State,ExitCode --noheader | grep -E '{failed_pattern}'"
                stdout, _, _ = execute_command(cmd, verbose=False)
                if stdout.strip():
                    print_warning(f"Failed jobs (showing up to 5):")
                    failed_lines = stdout.strip().split("\n")
                    for i, line in enumerate(failed_lines[:5]):  # Show at most 5 failed jobs
                        print_warning(f"  {line}")
                    if len(failed_lines) > 5:
                        print_warning(f"  ... and {len(failed_lines) - 5} more")

            # Check if all jobs are done
            if running_count == 0:
                print_success("All jobs have reached a finished state")
                break

            # Count various progress indicators
            progress_metrics = [
                ("Shards started", f'grep -l "processing shard" {log_pattern} | wc -l'),
                ("Models loading", f'grep -l "Starting to load model" {log_pattern} | wc -l'),
                ("Engines initialized", f'grep -l "init engine" {log_pattern} | wc -l'),
                ("Completed shards", f'grep -l "Processed prompts: 100%" {log_pattern} | wc -l'),
            ]

            for label, cmd in progress_metrics:
                stdout, _, _ = execute_command(cmd, verbose=False)
                count = int(stdout.strip())
                percentage = (count / num_shards) * 100
                print(f"  {label}: {count}/{num_shards} ({percentage:.1f}%)")

            # Wait before checking again
            time.sleep(watchdog_interval_min * 60)
            counter += 1
    except KeyboardInterrupt:
        print_warning("Monitoring interrupted. Job is still running.")
        return


def check_job_completion(job_id):
    """Check if all array jobs completed successfully and report detailed status."""
    print_header("Checking Job Completion")

    # Define job states
    failed_states = [
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "DEADLINE",
        "BOOT_FAIL",
        "SPECIAL_EXIT",
    ]

    # Get detailed job information
    cmd = f"sacct -j {job_id} --format=JobID%20,JobName,Elapsed,State,ExitCode --noheader"
    stdout, _, _ = execute_command(cmd)

    # Parse the output and count jobs by state
    lines = stdout.strip().split("\n") if stdout.strip() else []
    total_jobs = len(lines)
    completed_jobs = sum(1 for line in lines if "COMPLETED" in line)
    failed_jobs = sum(1 for line in lines if any(state in line for state in failed_states))

    # Print job statistics
    print_info(f"Total jobs: {total_jobs}")
    print_info(f"Completed jobs: {completed_jobs}")
    print_info(f"Failed jobs: {failed_jobs}")

    # Show detail on failed jobs if any
    if failed_jobs > 0:
        print_warning("Failed jobs:")
        failure_types = {}

        # Group failures by state
        for line in lines:
            if any(state in line for state in failed_states):
                for state in failed_states:
                    if state in line:
                        failure_types.setdefault(state, []).append(line)
                        break

        # Print summary by failure type
        for failure_type, failed_lines in failure_types.items():
            print_warning(f"  {failure_type}: {len(failed_lines)} jobs")

            # Print up to 3 examples of each failure type
            for i, line in enumerate(failed_lines[:3]):
                print_warning(f"    {line}")

            if len(failed_lines) > 3:
                print_warning(f"    ... and {len(failed_lines) - 3} more {failure_type} jobs")

    if completed_jobs == total_jobs:
        print_success("All jobs completed successfully.")
    else:
        print_warning(f"{completed_jobs}/{total_jobs} jobs completed successfully.")

    # Calculate and print time statistics for completed jobs
    if completed_jobs > 0:
        cmd = f"""sacct -j {job_id} -X --format=JobID%20,JobName,Elapsed,State --noheader | grep COMPLETED | awk '
        {{
            split($3, time, ":");
            seconds = time[1]*3600 + time[2]*60 + time[3];
            total += seconds;
            if (NR == 1 || seconds < min) min = seconds;
            if (NR == 1 || seconds > max) max = seconds;
            count++;
        }}
        END {{
            avg = total/count;
            printf "Min: %02d:%02d:%02d, Max: %02d:%02d:%02d, Mean: %02d:%02d:%02d\\n", 
                int(min/3600), int((min%3600)/60), min%60,
                int(max/3600), int((max%3600)/60), max%60,
                int(avg/3600), int((avg%3600)/60), avg%60
        }}'"""

        stdout, _, _ = execute_command(cmd)
        if stdout.strip():
            print_success(f"Job timing statistics (for completed jobs):\n  {stdout}")

    # Return true if enough jobs completed to consider the overall job successful
    # Here we're considering 90% completion as a reasonable threshold, but this could be adjusted
    return completed_jobs >= total_jobs * 0.9


def compute_and_upload_scores(tasks, output_repo_id, model_name):
    """Compute and upload scores."""
    print_header("Computing and Uploading Scores")
    print_warning(
        "This may take a while the first time the eval datasets are downloaded and parsed. Consider running locally with more cpus."
    )

    # replace / with __ in model_name
    model_name = model_name.replace("/", "__")
    tasks_str = ",".join(tasks)
    cmd = f'OPENAI_API_KEY=NONE python -m eval.eval --model precomputed_hf --model_args "repo_id={output_repo_id}",model="{model_name}" --tasks {tasks_str} --output_path logs --use_database'

    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_error(f"Failed to compute and upload scores: {stderr}")
        return False

    print_success("Scores computed and uploaded successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Distributed Evaluation Job Manager")
    parser.add_argument(
        "--tasks",
        type=str,
        default="LiveCodeBench,AIME24,AIME25,AMC23,GPQADiamond,MATH500",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name/path to evaluate")
    parser.add_argument("--num_shards", type=int, default=128, help="Number of shards for distributed evaluation")
    parser.add_argument("--watchdog", action="store_true", help="Monitor job progress and compute scores when done")

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Validate tasks
    tasks = [task.strip() for task in args.tasks.split(",")]
    print_info(f"Tasks to evaluate: {', '.join(tasks)}")

    # Check required environment variables
    if not check_required_env_vars():
        sys.exit(1)

    # Activate conda environment
    if not activate_conda_env():
        sys.exit(1)

    # Generate timestamp and repository ID for results
    timestamp = datetime.datetime.now().strftime("%m-%d-%y_%H-%M")
    model_name_short = args.model_name.split("/")[-1]
    task_hash = generate_task_hash(tasks)
    output_dataset = f"mlfoundations-dev/{model_name_short}_eval_{timestamp}_{task_hash}"

    print_info(f"Output dataset for results: {output_dataset}")

    # Create or get cached evaluation dataset
    input_dataset = create_evaluation_dataset(tasks)
    if not input_dataset:
        sys.exit(1)

    # Prepare for sbatch job
    logs_dir = prepare_for_sbatch(input_dataset, output_dataset, args.model_name)

    # Launch sbatch job with the dataset repo but save to output repo
    job_id = launch_sbatch(args.model_name, input_dataset, output_dataset, args.num_shards, logs_dir, args.tasks)
    if not job_id:
        sys.exit(1)

    # Print helpful commands
    print_header("Helpful Commands")
    print_info(f"Monitor job status: squeue -j {job_id}")
    print_info(f"Cancel job: scancel {job_id}")
    print_info(f"View detailed job info: sacct -j {job_id} -X --format=JobID,JobName,State,Elapsed")
    print_info(f"Monitor logs: tail -f {logs_dir}/{job_id}_*.out")
    print_info(f"Watch shards get uploaded: https://huggingface.co/datasets/{output_dataset}/tree/main")

    # If watchdog flag is not set, exit
    if not args.watchdog:
        print_info("Watchdog mode not enabled. Exiting.")
        return

    # Monitor job
    print_info("Watchdog mode enabled. Monitoring job progress...")
    monitor_job(job_id, logs_dir, args.num_shards)

    # Check job completion
    success = check_job_completion(job_id)

    # Compute and upload scores
    if success:
        if compute_and_upload_scores(tasks, output_dataset, args.model_name):
            print_success(f"Evaluation completed successfully. Results uploaded to {output_dataset}")
            print_info(f"View the results at: https://huggingface.co/{output_dataset}")
            print_info(f"Dataset used: {input_dataset}")
        else:
            print_error("Failed to compute and upload scores.")
    else:
        print_warning("Some jobs failed. You may want to check the logs before computing scores.")
        response = input("Would you like to compute scores anyway? (y/n): ")
        if response.lower() == "y":
            compute_and_upload_scores(tasks, output_dataset, args.model_name)
            print_info(f"Dataset used: {input_dataset}")


if __name__ == "__main__":
    main()
