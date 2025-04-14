#!/usr/bin/env python

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download


def execute_command(cmd, env=None, verbose=True):
    """Execute a shell command and return the output."""
    if verbose:
        print(f"Running: {cmd}")

    if env is None:
        env = os.environ.copy()

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True
    )

    stdout, stderr = process.communicate()
    return_code = process.returncode
    if verbose:
        print(f"Command failed with return code {return_code}")
        print(f"Error: {stderr.strip()}")

    return stdout.strip(), stderr.strip(), return_code


def generate_evaluation_dataset_hash(tasks, system_instruction=None):
    """Generate a 4-character hash from the task list and system instruction."""
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


def create_evaluation_dataset(tasks, system_instruction=None):
    """Create or use cached evaluation dataset."""

    # Generate a cached dataset name based on tasks
    eval_dataset_hash = generate_evaluation_dataset_hash(tasks, system_instruction)
    cached_dataset_id = f"mlfoundations-dev/evalset_{eval_dataset_hash}"

    # Check if the cached dataset exists
    if check_dataset_exists(cached_dataset_id):
        print(f"Using cached evaluation dataset: {cached_dataset_id}")
        return cached_dataset_id

    # If not, create a new evaluation dataset
    print("Creating new evaluation dataset...")
    print("Coding tasks require heavier processing to generate prompts and may take a little while...")
    tasks_str = ",".join(tasks)
    if system_instruction:
        cmd = f"python -m eval.eval --model upload_to_hf --tasks {tasks_str} --model_args repo_id={cached_dataset_id} --output_path logs --system_instruction '{system_instruction}'"
    else:
        cmd = f"python -m eval.eval --model upload_to_hf --tasks {tasks_str} --model_args repo_id={cached_dataset_id} --output_path logs"

    print(f"Running command: {cmd}")
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print("Failed to create evaluation dataset.")
        print(f"Error: {stderr}")
        return False

    print(f"Evaluation dataset created at: https://huggingface.co/datasets/{cached_dataset_id}")
    return cached_dataset_id


def download_model(model_name):
    """Download a model from Hugging Face."""
    print(f"Downloading model: {model_name}")
    try:
        model_path = snapshot_download(repo_id=model_name)
        print(f"Model downloaded successfully to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Failed to download model: {str(e)}")
        sys.exit(1)


def download_dataset(dataset_name):
    """Download a dataset from Hugging Face."""
    print(f"Downloading dataset: {dataset_name}")
    try:
        dataset_path = snapshot_download(repo_id=dataset_name, repo_type="dataset")
        print(f"Dataset downloaded successfully to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Failed to download dataset: {str(e)}")
        sys.exit(1)


def launch_eval_sbatch(cmd, logs_dir, sbatch_script):
    """Launch the sbatch job for evaluation step."""
    # Create a temporary sbatch script with the correct parameters
    temp_sbatch_file = os.path.join(logs_dir, "job.sbatch")
    with open(sbatch_script, "r") as f:
        sbatch_content = f.read()

    sbatch_content = re.sub(r"export EVAL_COMMAND=.*", f'export EVAL_COMMAND="{cmd}"', sbatch_content)
    sbatch_content = re.sub(r"(^#!.*\n)", r"\1#SBATCH --output=" + logs_dir + r"/%A_%a.out\n", sbatch_content)

    with open(temp_sbatch_file, "w") as f:
        f.write(sbatch_content)

    print(f"Created temporary sbatch file: {temp_sbatch_file}")

    # Launch the sbatch job
    cmd = f"sbatch {temp_sbatch_file}"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print(f"Failed to launch sbatch job: {stderr}")
        return None, None

    # Extract the job ID from the output
    job_id_match = re.search(r"Submitted batch job (\d+)", stdout)
    if job_id_match:
        job_id = job_id_match.group(1)
        print(f"SBATCH job submitted with ID: {job_id}")
    else:
        print("Could not determine job ID from sbatch output.")
        job_id = None

    print(f"[Job status] squeue -j {job_id}")
    print(f"[Job status] sacct -j {job_id} -X --format=JobID,JobName,State,Elapsed")
    print(f"[Cancel job] scancel {job_id}")
    print(f"[View logs] tail {logs_dir}/{job_id}_*.out")

    return job_id


def launch_sbatch(
    model_path,
    dataset_path,
    output_dataset_dir,
    num_shards,
    logs_dir,
    max_job_duration=None,
    tp4=False,
):
    """Launch the sbatch job."""

    # Check hostname to determine which sbatch script to use
    cmd = "echo $HOSTNAME"
    hostname, _, _ = execute_command(cmd, verbose=False)
    print(f"Using $HOSTNAME: {hostname} to determine which sbatch script to use")
    if "c1" in hostname or "c2" in hostname:
        sbatch_script = "eval/distributed/process_shards_capella.sbatch"
    elif "leonardo" in hostname:
        sbatch_script = "eval/distributed/process_shards_leonardo.sbatch"
    elif "tacc" in hostname:
        sbatch_script = "eval/distributed/process_shards_tacc.sbatch"
    else:
        raise ValueError(f"Unknown hostname: {hostname}, can't determine which sbatch script to use")

    # Create a temporary sbatch script with the correct parameters
    temp_sbatch_file = os.path.join(logs_dir, "job.sbatch")
    with open(sbatch_script, "r") as f:
        sbatch_content = f.read()

    # Replace parameters in the sbatch script using regex pattern matching
    sbatch_content = re.sub(r"#SBATCH --array=.*", f"#SBATCH --array=0-{num_shards-1}", sbatch_content)
    sbatch_content = re.sub(r"export INPUT_DATASET=.*", f'export INPUT_DATASET="{dataset_path}"', sbatch_content)
    sbatch_content = re.sub(
        r"export OUTPUT_DATASET=.*", f'export OUTPUT_DATASET="{output_dataset_dir}"', sbatch_content
    )
    sbatch_content = re.sub(r"export MODEL_NAME=.*", f'export MODEL_NAME="{model_path}"', sbatch_content)
    sbatch_content = re.sub(r"(^#!.*\n)", r"\1#SBATCH --output=" + logs_dir + r"/%A_%a.out\n", sbatch_content)

    # Update job duration if specified
    if max_job_duration:
        formatted_duration = f"{max_job_duration:02d}:00:00"
        sbatch_content = re.sub(r"#SBATCH --time=.*", f"#SBATCH --time={formatted_duration}", sbatch_content)
        print(f"Setting job duration to {formatted_duration}")

    with open(temp_sbatch_file, "w") as f:
        f.write(sbatch_content)

    print(f"Created temporary sbatch file: {temp_sbatch_file}")

    # Launch the sbatch job
    cmd = f"sbatch {temp_sbatch_file}"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print(f"Failed to launch sbatch job: {stderr}")
        return None, None

    # Extract the job ID from the output
    job_id_match = re.search(r"Submitted batch job (\d+)", stdout)
    if job_id_match:
        job_id = job_id_match.group(1)
        print(f"SBATCH job submitted with ID: {job_id}")
    else:
        print("Could not determine job ID from sbatch output.")
        job_id = None

    print(f"Results will be saved locally to {output_dataset_dir}")
    print(f"[Job status] squeue -j {job_id}")
    print(f"[Job status] sacct -j {job_id} -X --format=JobID,JobName,State,Elapsed")
    print(f"[Cancel job] scancel {job_id}")
    print(f"[View logs] tail {logs_dir}/{job_id}_*.out")

    return job_id


def upload_shards_to_hub(output_dir, output_repo_id):
    """Upload all locally saved shards to HuggingFace Hub."""

    # Check if output directory exists using shell command
    cmd = f"test -d {output_dir} && echo 'exists' || echo 'not exists'"
    stdout, _, _ = execute_command(cmd)
    if stdout.strip() == "not exists":
        print(f"Output directory {output_dir} does not exist")
        return False

    # Check if there are any parquet files
    cmd = f"ls -1 {output_dir}/*.parquet 2>/dev/null | wc -l"
    stdout, _, _ = execute_command(cmd)
    file_count = int(stdout.strip())

    if file_count == 0:
        print(f"No parquet files found in {output_dir}")
        return False

    print(f"Found {file_count} parquet files to upload")

    # Parse repository ID to get organization and repository name
    parts = output_repo_id.split("/")
    if len(parts) != 2:
        print(f"Invalid repository ID format: {output_repo_id}. Expected format: 'organization/repository'")
        return False

    org = parts[0]
    repo_name = parts[1]

    # Create the dataset repository if it doesn't exist
    cmd = f"huggingface-cli repo create {repo_name} --organization {org} --type dataset -y || echo 'Repository already exists'"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print(f"Repository creation returned non-zero status: {stderr}")

    # Upload all files
    print(f"Uploading files from {output_dir} to {output_repo_id}...")
    cmd = f"huggingface-cli upload {output_repo_id} {output_dir} --repo-type dataset"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print(f"Failed to upload files: {stderr}")
        return False

    print(f"All files successfully uploaded to {output_repo_id}")
    print(f"View the dataset at https://huggingface.co/datasets/{output_repo_id}")
    return True


def compute_and_upload_scores(tasks, output_repo_id, model_name, logs_dir, on_login):
    """Compute and upload scores."""
    if "LiveCodeBench" in tasks:
        print("LiveCodeBench evaluation takes ~15mins")

    tasks_str = ",".join(tasks)
    cmd = f'python -m eval.eval --model precomputed_hf --model_args "repo_id={output_repo_id}",model="{model_name}" --tasks {tasks_str} --output_path logs --use_database'

    # Check hostname to determine which sbatch script to use
    hostname_cmd = "echo $HOSTNAME"
    hostname, _, _ = execute_command(hostname_cmd, verbose=False)
    print(f"Using $HOSTNAME: {hostname} to determine cluster environment.")
    if not on_login and ("tacc" in hostname or "c1" in hostname or "c2" in hostname):
        if "tacc" in hostname:
            sbatch_script = "eval/distributed/run_evaluations_tacc.sbatch"
        elif "c1" in hostname or "c2" in hostname:
            sbatch_script = "eval/distributed/run_evaluations_capella.sbatch"
        print("Computing scores on node")
        job_id = launch_eval_sbatch(cmd, logs_dir, sbatch_script)
        if not job_id:
            return False

        print("Watchdog mode enabled. Monitoring job progress...")
        monitor_job(job_id, logs_dir, 1)

        # Check completion
        if not check_job_completion(job_id):
            print("Some jobs failed. Failed to compute and upload scores.")
            return False
    else:
        print("Computing scores on login node")
        stdout, stderr, return_code = execute_command(cmd)

        if return_code != 0:
            print(f"Failed to compute and upload scores: {stderr}")
            return False

    print("Scores computed and uploaded successfully.")
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
    parser.add_argument(
        "--max-job-duration",
        type=int,
        default=None,
        help="Maximum job duration in hours (default: use sbatch script default)",
    )
    parser.add_argument("--system_instruction", type=str, default=None, help="System instruction for the model")
    parser.add_argument("--timestamp", action="store_true", help="Add a timestamp to the output evaluation dataset")

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Validate tasks
    tasks = [task.strip() for task in args.tasks.split(",")]
    print(f"Tasks to evaluate: {', '.join(tasks)}")

    # Generate timestamp and repository ID for results
    evaluation_dataset_hash = generate_evaluation_dataset_hash(tasks, args.system_instruction)
    if args.timestamp:
        timestamp = str(int(time.time()))
        suffix = f"_{timestamp}_eval_{evaluation_dataset_hash}"
    else:
        suffix = f"_eval_{evaluation_dataset_hash}"
    remaining_characters = 96 - len(suffix)
    model_name_short = args.model_name.split("/")[-1][:remaining_characters]
    output_dataset = f"mlfoundations-dev/{model_name_short}{suffix}"

    # Create or get cached evaluation dataset
    input_dataset = create_evaluation_dataset(tasks, args.system_instruction)
    if not input_dataset:
        sys.exit(1)

    # Output directories
    output_dataset_repo_name = output_dataset.split("/")[-1]
    logs_dir = os.path.join("logs", output_dataset_repo_name)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Logs directory: {logs_dir}")
    output_dataset_dir = os.path.join("results", output_dataset_repo_name)
    os.makedirs(output_dataset_dir, exist_ok=True)
    print(f"Output dataset directory: {output_dataset_dir}")

    # Download the dataset and model
    dataset_path = download_dataset(input_dataset)
    model_path = download_model(args.model_name)

    # Launch sbatch job with the dataset repo but save to output repo
    job_id = launch_sbatch(
        model_path,
        dataset_path,
        output_dataset_dir,
        args.num_shards,
        logs_dir,
        args.max_job_duration,
        args.tp4,
    )
    if not job_id:
        sys.exit(1)

    # If watchdog flag is not set, exit
    if not args.watchdog:
        print("Watchdog mode not enabled. Exiting.")
        exit(0)

    # Monitor job
    print("Watchdog mode enabled. Monitoring job progress...")
    monitor_job(job_id, logs_dir, args.num_shards)

    # Check completion
    if not check_job_completion(job_id, output_dataset_dir):
        print("Some jobs failed.")
        exit(1)

    # Upload shards
    upload_shards_to_hub(output_dataset_dir, output_dataset)

    # Compute and upload scores
    if compute_and_upload_scores(tasks, output_dataset, args.model_name, logs_dir, on_login=args.on_login):
        print(f"Evaluation completed successfully. Results uploaded to {output_dataset}")
        print(f"View the results at: https://huggingface.co/datasets/{output_dataset}")
    else:
        print("Failed to compute and upload scores.")
        exit(1)


if __name__ == "__main__":
    main()
