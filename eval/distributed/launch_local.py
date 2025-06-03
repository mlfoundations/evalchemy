#!/usr/bin/env python

import argparse
import datetime
import hashlib
import os
import re
import subprocess
import sys
import time

import colorama
from colorama import Fore, Style
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

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

    # If env is None, copy the current environment including HF_HUB
    if env is None:
        env = os.environ.copy()

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, universal_newlines=True
    )

    stdout, stderr = process.communicate()
    return_code = process.returncode

    if return_code != 0 and verbose:
        print_error(f"Command failed with return code {return_code}")
        print_error(f"Error: {stderr.strip()}")

    return stdout.strip(), stderr.strip(), return_code


def check_required_env_vars(mode="auto"):
    """Check if required environment variables are set."""
    print_header("Checking Environment Variables")

    # Determine if we're in local mode
    if mode == "auto":
        cmd = "echo $HOSTNAME"
        hostname, _, _ = execute_command(cmd, verbose=False)
        is_local = not ("c1" in hostname or "leonardo" in hostname)
    else:
        is_local = mode == "local"

    required_vars = ["HF_TOKEN", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER"]
    missing_vars = []
    for var in required_vars:
        if os.environ.get(var) is None:
            missing_vars.append(var)

    # Check hostname to determine which HF_HUB_CACHE to use
    cmd = "echo $HOSTNAME"
    hostname, _, _ = execute_command(cmd, verbose=False)
    print_info(f"Using $HOSTNAME: {hostname} to determine which HF_HUB_CACHE to use")
    if "c1" in hostname or "c2" in hostname:
        hf_hub_cache = "/data/horse/ws/ryma833h-DCFT_Shared/huggingface/hub"
        print_info(f"Detected Capella environment, using HF_HUB_CACHE: {hf_hub_cache}")
    elif "leonardo" in hostname:
        hf_hub_cache = "/leonardo_work/EUHPC_E03_068/DCFT_shared/hub"
        print_info(f"Detected Leonardo environment, using HF_HUB_CACHE: {hf_hub_cache}")
    elif "tacc" in hostname:
        hf_hub_cache = "/scratch/08134/negin/hf_home/"
        print_info(f"Detected TACC environment, using HF_HUB_CACHE: {hf_hub_cache}")
    else:
        # Use the default cache location for local environments
        home_dir = os.path.expanduser("~")
        hf_hub_cache = os.path.join(home_dir, ".cache/huggingface/hub")
        print_info(f"No specific cluster detected, using default local HF_HUB_CACHE: {hf_hub_cache}")
        # Ensure the directory exists
        os.makedirs(hf_hub_cache, exist_ok=True)

    current_hub_cache = os.environ.get("HF_HUB_CACHE")
    if current_hub_cache is not None and current_hub_cache != hf_hub_cache:
        print_warning(f"Overwriting existing HF_HUB_CACHE value '{current_hub_cache}' with '{hf_hub_cache}'")
    os.environ["HF_HUB_CACHE"] = hf_hub_cache

    if missing_vars:
        print_error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print_info("Please set these variables using one of the following methods:")
        print_info("1. Add them to your shell profile (~/.bashrc) and restart your shell")
        print_info("2. Export them in your current shell: export HF_TOKEN=your_token")
        print_info(
            "3. Set them in .env file in evalchemy root (not recommended for sensitive tokens on shared systems)"
        )
        if not is_local:
            print_info("")
            print_info("IMPORTANT: These variables will be automatically passed to worker nodes via SLURM's")
            print_info("environment propagation (e.g. #SBATCH --export=ALL) which is the default behavior.")
        return False

    print_success("All required environment variables are set.")
    return True


def check_conda_env(mode="auto", watchdog=False):
    """Check if the conda environment is activated."""
    print_header("Checking Conda Environment")

    # Check if we're in local mode
    if mode == "auto":
        cmd = "echo $HOSTNAME"
        hostname, _, _ = execute_command(cmd, verbose=False)
        is_local = not ("c1" in hostname or "leonardo" in hostname)
    else:
        is_local = mode == "local"

    # Check hostname to determine which conda environment we should be in
    cmd = "echo $HOSTNAME"
    hostname, _, _ = execute_command(cmd, verbose=False)
    print_info(f"Using $HOSTNAME: {hostname} to determine which conda environment we should be in")
    if "c1" in hostname or "c2" in hostname:
        python_path = "/data/horse/ws/ryma833h-DCFT_Shared/evalchemy/env/evalchemy/bin/python3.10"
        activate_cmd = "source /data/horse/ws/ryma833h-DCFT_Shared/miniconda3/bin/activate /data/horse/ws/ryma833h-DCFT_Shared/evalchemy/env/evalchemy"
        print_info(f"Detected Capella environment, checking python path: {python_path}")
    elif "leonardo" in hostname:
        python_path = "/leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/env/cpu-evalchemy/bin/python3.10"
        activate_cmd = "source /leonardo_work/EUHPC_E03_068/DCFT_shared/mamba/bin/activate /leonardo_work/EUHPC_E03_068/DCFT_shared/evalchemy/env/cpu-evalchemy"
        print_info(f"Detected Leonardo environment, checking python path: {python_path}")
    elif "tacc" in hostname:
        python_path = "/work/08134/negin/anaconda3/envs/evalchemy/bin/python3.10"
        activate_cmd = "source /work/08134/negin/anaconda3/bin/activate evalchemy"
        print_info(f"Detected TACC environment, checking python path: {python_path}")
    else:
        # For local environments, we don't enforce a specific Python path
        print_info(f"Local environment detected, not enforcing specific conda environment")

        # If watchdog is enabled, just check if Python is accessible
        if watchdog and is_local:
            cmd = "which python"
            stdout, _, return_code = execute_command(cmd)
            if return_code != 0:
                print_error("Python not found. Please make sure Python is installed and available in your PATH.")
                return False
            print_info(f"Using Python from: {stdout}")

        return True

    # we'll check if the environment exists
    cmd = "readlink -f $(which python)"
    stdout, _, return_code = execute_command(cmd)

    if return_code != 0 or stdout != python_path:
        if watchdog:
            print_error(f"You have not activated the correct conda environment. Your python path is: {stdout}")
            print_error("This is necessary when '--watchdog' is used. Please activate the evalchemy conda environment:")
            print_error(activate_cmd)
            return False
        else:
            print_warning(f"You are not using the suggested conda environment. Your python path is: {stdout}")

    return True


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
    print_header("Preparing Evaluation Dataset")

    # Generate a cached dataset name based on tasks
    eval_dataset_hash = generate_evaluation_dataset_hash(tasks, system_instruction)
    cached_dataset_id = f"mlfoundations-dev/evalset_{eval_dataset_hash}"

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
    if system_instruction:
        cmd = f"python -m eval.eval --model upload_to_hf --tasks {tasks_str} --model_args repo_id={cached_dataset_id} --output_path logs --system_instruction '{system_instruction}'"
    else:
        cmd = f"python -m eval.eval --model upload_to_hf --tasks {tasks_str} --model_args repo_id={cached_dataset_id} --output_path logs"

    print_warning(f"Running command: {cmd}")
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_error("Failed to create evaluation dataset.")
        print_error(f"Error: {stderr}")
        return False

    print_success(f"Evaluation dataset created at: https://huggingface.co/datasets/{cached_dataset_id}")
    return cached_dataset_id


def download_model(model_name):
    """Download a model from Hugging Face."""
    print_info(f"Downloading model: {model_name}")
    try:
        model_path = snapshot_download(repo_id=model_name)
        print_success(f"Model downloaded successfully to: {model_path}")
        return model_path
    except Exception as e:
        print_error(f"Failed to download model: {str(e)}")
        sys.exit(1)


def download_dataset(dataset_name):
    """Download a dataset from Hugging Face."""
    print_info(f"Downloading dataset: {dataset_name}")
    try:
        dataset_path = snapshot_download(repo_id=dataset_name, repo_type="dataset")
        print_success(f"Dataset downloaded successfully to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print_error(f"Failed to download dataset: {str(e)}")
        sys.exit(1)


def launch_local(
    model_path,
    dataset_path,
    output_dataset_dir,
    num_shards,
    logs_dir,
    tp4,
):
    """Launch local job processing using the local shell script."""
    print_header("Launching Local Processing Job")

    if not tp4:
        local_script = "eval/distributed/process_shards_local.sh"
    else:
        local_script = "eval/distributed/process_shards_local_tp.sh"
    print_info("Using process_shards_local.sh for local processing")

    # Create a temporary script file with the correct parameters
    temp_script_file = os.path.join(logs_dir, "job_local.sh")
    with open(local_script, "r") as f:
        script_content = f.read()

    # Replace parameters in the script using regex pattern matching
    script_content = re.sub(r"export GLOBAL_SIZE=.*", f"export GLOBAL_SIZE={num_shards}", script_content)
    script_content = re.sub(r"export MODEL_NAME=.*", f'export MODEL_NAME="{model_path}"', script_content)
    script_content = re.sub(r"export INPUT_DATASET=.*", f'export INPUT_DATASET="{dataset_path}"', script_content)
    script_content = re.sub(
        r"export OUTPUT_DATASET=.*", f'export OUTPUT_DATASET="{output_dataset_dir}"', script_content
    )

    # Update the GPU range in the for loop based on num_shards
    gpu_range_pattern = r"for RANK in \{.*\}; do"
    gpu_range = f"for RANK in {{0..{min(num_shards-1, 7)}}}; do"  # Limit to 8 GPUs by default
    script_content = re.sub(gpu_range_pattern, gpu_range, script_content)

    with open(temp_script_file, "w") as f:
        f.write(script_content)

    print_success(f"Created temporary script file: {temp_script_file}")
    os.chmod(temp_script_file, 0o755)  # Make it executable

    # Launch the local job
    job_id = f"local_{int(time.time())}"
    log_file = os.path.join(logs_dir, f"{job_id}.out")
    cmd = f"bash {temp_script_file} > {log_file} 2>&1 &"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_error(f"Failed to launch local job: {stderr}")
        return None

    # Get the process ID of the background job
    cmd = "echo $!"
    stdout, _, _ = execute_command(cmd)
    if stdout.strip():
        process_id = stdout.strip()
        print_success(f"Local job launched with process ID: {process_id}")
    else:
        process_id = "unknown"
        print_warning("Could not determine process ID for local job")

    print_info(f"Results will be saved locally to {output_dataset_dir}")
    print_info(f"[View logs] tail -f {log_file}")

    return job_id


def launch_eval_sbatch(cmd, logs_dir):
    """Launch the sbatch job for evaluation step."""
    print_header("Launching SBATCH Job")

    sbatch_script = "eval/distributed/run_evaluations_tacc.sbatch"
    # Create a temporary sbatch script with the correct parameters
    temp_sbatch_file = os.path.join(logs_dir, "job.sbatch")
    with open(sbatch_script, "r") as f:
        sbatch_content = f.read()

    sbatch_content = re.sub(r"export EVAL_COMMAND=.*", f'export EVAL_COMMAND="{cmd}"', sbatch_content)
    sbatch_content = re.sub(r"(^#!.*\n)", r"\1#SBATCH --output=" + logs_dir + r"/%A_%a.out\n", sbatch_content)

    with open(temp_sbatch_file, "w") as f:
        f.write(sbatch_content)

    print_success(f"Created temporary sbatch file: {temp_sbatch_file}")

    # Launch the sbatch job
    cmd = f"sbatch {temp_sbatch_file}"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_error(f"Failed to launch sbatch job: {stderr}")
        return None, None

    # Extract the job ID from the output
    job_id_match = re.search(r"Submitted batch job (\d+)", stdout)
    if job_id_match:
        job_id = job_id_match.group(1)
        print_success(f"SBATCH job submitted with ID: {job_id}")
    else:
        print_error("Could not determine job ID from sbatch output.")
        job_id = None

    print_info(f"[Job status] squeue -j {job_id}")
    print_info(f"[Job status] sacct -j {job_id} -X --format=JobID,JobName,State,Elapsed")
    print_info(f"[Cancel job] scancel {job_id}")
    print_info(f"[View logs] tail {logs_dir}/{job_id}_*.out")

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
    print_header("Launching SBATCH Job")

    # Check hostname to determine which sbatch script to use
    cmd = "echo $HOSTNAME"
    hostname, _, _ = execute_command(cmd, verbose=False)
    print_info(f"Using $HOSTNAME: {hostname} to determine which sbatch script to use")
    if "c1" in hostname or "c2" in hostname:
        if tp4:
            sbatch_script = "eval/distributed/process_shards_capella_tp.sbatch"
            print_info("Detected Capella environment with TP4, using process_shards_capella_tp.sbatch")
        else:
            sbatch_script = "eval/distributed/process_shards_capella.sbatch"
            print_info("Detected Capella environment, using process_shards_capella.sbatch")
    elif "leonardo" in hostname:
        sbatch_script = "eval/distributed/process_shards_leonardo.sbatch"
        print_info("Detected Leonardo environment, using process_shards_leonardo.sbatch")
        if tp4:
            sbatch_script = "eval/distributed/process_shards_leonardo_tp.sbatch"
            print_info("Detected Leonardo environment with TP4, using process_shards_leonardo_tp.sbatch")
        else:
            sbatch_script = "eval/distributed/process_shards_leonardo.sbatch"
            print_info("Detected Leonardo environment, using process_shards_leonardo.sbatch")
    elif "tacc" in hostname:
        sbatch_script = "eval/distributed/process_shards_tacc.sbatch"
        print_info("Detected TACC environment, using process_shards_tacc.sbatch")
    else:
        print_warning(f"Unknown hostname: {hostname}, defaulting to local processing")
        return launch_local(model_path, dataset_path, output_dataset_dir, num_shards, logs_dir, tp4)

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
        print_info(f"Setting job duration to {formatted_duration}")

    with open(temp_sbatch_file, "w") as f:
        f.write(sbatch_content)

    print_success(f"Created temporary sbatch file: {temp_sbatch_file}")

    # Launch the sbatch job
    cmd = f"sbatch {temp_sbatch_file}"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_error(f"Failed to launch sbatch job: {stderr}")
        return None, None

    # Extract the job ID from the output
    job_id_match = re.search(r"Submitted batch job (\d+)", stdout)
    if job_id_match:
        job_id = job_id_match.group(1)
        print_success(f"SBATCH job submitted with ID: {job_id}")
    else:
        print_error("Could not determine job ID from sbatch output.")
        job_id = None

    print_info(f"Results will be saved locally to {output_dataset_dir}")
    print_info(f"[Job status] squeue -j {job_id}")
    print_info(f"[Job status] sacct -j {job_id} -X --format=JobID,JobName,State,Elapsed")
    print_info(f"[Cancel job] scancel {job_id}")
    print_info(f"[View logs] tail {logs_dir}/{job_id}_*.out")

    return job_id


def monitor_local_job(job_id, logs_dir, num_shards, watchdog_interval_min=1):
    """Monitor a local job's progress."""
    print_header("Monitoring Local Job Progress")

    # Determine the log file pattern based on the job ID
    log_file = f"{logs_dir}/{job_id}.out"

    # Make sure there's enough time for jobs to start
    time.sleep(5)

    counter = 0
    try:
        while True:
            # Check if the job is still running by checking for process activity
            cmd = "ps aux | grep process_shard.py | grep -v grep | wc -l"
            stdout, _, _ = execute_command(cmd, verbose=False)
            running_count = int(stdout.strip())

            # Count various progress indicators
            progress_metrics = [
                ("Shards started", f'grep -c "processing shard" {log_file}'),
                ("Models loading", f'grep -c "Starting to load model" {log_file}'),
                ("Engines initialized", f'grep -c "init engine" {log_file}'),
                ("Completed shards", f'grep -c "Shard successfully processed" {log_file}'),
            ]

            results = {}
            for label, cmd in progress_metrics:
                stdout, _, _ = execute_command(cmd, verbose=False)
                count = int(stdout.strip()) if stdout.strip().isdigit() else 0
                results[label] = count

            print_info(
                f"({counter*watchdog_interval_min}m) Local Job Status: {results['Completed shards']} completed, {running_count} processes running"
            )

            for label, count in results.items():
                percentage = (count / num_shards) * 100
                print(f"  {label}: {count}/{num_shards} ({percentage:.1f}%)")

            # Check if all shards are completed
            if results["Completed shards"] >= num_shards or running_count == 0:
                if results["Completed shards"] >= num_shards:
                    print_success("All shards have been processed")
                else:
                    print_warning("No processes running but not all shards completed")
                break

            # Wait before checking again
            time.sleep(watchdog_interval_min * 60)
            counter += 1
    except KeyboardInterrupt:
        print_warning("Monitoring interrupted. Job may still be running.")
        return


def monitor_job(job_id, logs_dir, num_shards, watchdog_interval_min=1):
    """Monitor the slurm job and show progress."""
    # Check if this is a local job
    if job_id.startswith("local_"):
        return monitor_local_job(job_id, logs_dir, num_shards, watchdog_interval_min)

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
                ("Completed shards", f'grep -l "Shard successfully processed" {log_pattern} | wc -l'),
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


def check_local_job_completion(job_id, output_dir=None):
    """Check if a local job completed successfully."""
    print_header("Checking Local Job Completion")

    # Check if the output directory contains parquet files
    if output_dir:
        cmd = f"ls -1 {output_dir}/*.parquet 2>/dev/null | wc -l"
        stdout, _, _ = execute_command(cmd)
        file_count = int(stdout.strip())
        print_info(f"Found {file_count} parquet files in {output_dir}")

        # Check the log file for errors
        log_file = f"logs/{output_dir.split('/')[-1]}/{job_id}.out"
        cmd = f"grep -c 'ERROR' {log_file}"
        stdout, _, _ = execute_command(cmd)
        error_count = int(stdout.strip()) if stdout.strip().isdigit() else 0

        if error_count > 0:
            print_warning(f"Found {error_count} errors in log file")
            # Show a sample of errors
            cmd = f"grep 'ERROR' {log_file} | head -5"
            stdout, _, _ = execute_command(cmd)
            if stdout.strip():
                print_warning("Sample errors:")
                for line in stdout.strip().split("\n"):
                    print_warning(f"  {line}")

        # Return true if we have a reasonable number of parquet files
        # This is a heuristic - adjust based on your needs
        return file_count > 0

    return False


def check_job_completion(job_id, output_dir=None):
    """Check if all array jobs completed successfully and report detailed status."""
    # Check if this is a local job
    if job_id.startswith("local_"):
        return check_local_job_completion(job_id, output_dir)

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
    cmd = f"sacct -j {job_id} -X --format=JobID%20,JobName,Elapsed,State,ExitCode --noheader"
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

            # Special message for TIMEOUT failures
            if failure_type == "TIMEOUT":
                print_error("Jobs timed out! Use --max-job-duration parameter to increase the time limit.")

    if completed_jobs == total_jobs:
        print_success("All jobs completed successfully.")
    else:
        print_warning(f"{completed_jobs}/{total_jobs} jobs completed successfully.")

    if output_dir:
        cmd = f"ls -1 {output_dir}/*.parquet 2>/dev/null | wc -l"
        stdout, _, _ = execute_command(cmd)
        file_count = int(stdout.strip())
        print_info(f"Found {file_count} parquet files in {output_dir}")

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
            total_hours = total / 3600;
            printf "Min: %02d:%02d:%02d, Max: %02d:%02d:%02d, Mean: %02d:%02d:%02d, Total: %.1f GPU hours\\n", 
                int(min/3600), int((min%3600)/60), min%60,
                int(max/3600), int((max%3600)/60), max%60,
                int(avg/3600), int((avg%3600)/60), avg%60,
                total_hours
        }}'"""

        stdout, _, _ = execute_command(cmd)
        if stdout.strip():
            print_success(f"Job timing statistics (for completed jobs):\n  {stdout}")

    # Return true if enough jobs completed to consider the overall job successful
    # Here we're considering 90% completion as a reasonable threshold, but this could be adjusted
    return completed_jobs >= total_jobs * 0.9


def upload_shards_to_hub(output_dir, output_repo_id):
    """Upload all locally saved shards to HuggingFace Hub."""
    print_header("Uploading Results to HuggingFace Hub")

    # Check if output directory exists using shell command
    cmd = f"test -d {output_dir} && echo 'exists' || echo 'not exists'"
    stdout, _, _ = execute_command(cmd)
    if stdout.strip() == "not exists":
        print_error(f"Output directory {output_dir} does not exist")
        return False

    # Check if there are any parquet files
    cmd = f"ls -1 {output_dir}/*.parquet 2>/dev/null | wc -l"
    stdout, _, _ = execute_command(cmd)
    file_count = int(stdout.strip())

    if file_count == 0:
        print_error(f"No parquet files found in {output_dir}")
        return False

    print_info(f"Found {file_count} parquet files to upload")

    # Parse repository ID to get organization and repository name
    parts = output_repo_id.split("/")
    if len(parts) != 2:
        print_error(f"Invalid repository ID format: {output_repo_id}. Expected format: 'organization/repository'")
        return False

    org = parts[0]
    repo_name = parts[1]

    # Create the dataset repository if it doesn't exist
    cmd = f"huggingface-cli repo create {repo_name} --organization {org} --type dataset -y || echo 'Repository already exists'"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_warning(f"Repository creation returned non-zero status: {stderr}")

    # Upload all files
    print_info(f"Uploading files from {output_dir} to {output_repo_id}...")
    cmd = f"huggingface-cli upload {output_repo_id} {output_dir} --repo-type dataset"
    stdout, stderr, return_code = execute_command(cmd)

    if return_code != 0:
        print_error(f"Failed to upload files: {stderr}")
        return False

    print_success(f"All files successfully uploaded to {output_repo_id}")
    print_info(f"View the dataset at https://huggingface.co/datasets/{output_repo_id}")
    return True


def compute_and_upload_scores(tasks, output_repo_id, model_name, logs_dir, use_database=True):
    """Compute and upload scores."""
    print_header("Computing and Uploading Scores")
    if "LiveCodeBench" in tasks:
        print_warning("LiveCodeBench evaluation takes ~15mins")

    tasks_str = ",".join(tasks)
    db_flag = "--use_database" if use_database else ""
    cmd = f'python -m eval.eval --model precomputed_hf --model_args "repo_id={output_repo_id}",model="{model_name}" --tasks {tasks_str} --output_path logs {db_flag}'

    # Check hostname to determine which sbatch script to use
    hostname_cmd = "echo $HOSTNAME"
    hostname, _, _ = execute_command(hostname_cmd, verbose=False)
    print_info(f"Using $HOSTNAME: {hostname} to determine cluster environment.")
    if "tacc" in hostname:
        print_info("Detected TACC environment. Computing scores on TACC.")
        job_id = launch_eval_sbatch(cmd, logs_dir)
        if not job_id:
            return False

        print_info("Watchdog mode enabled. Monitoring job progress...")
        monitor_job(job_id, logs_dir, 1)

        # Check completion
        if not check_job_completion(job_id):
            print_error("Some jobs failed. Failed to compute and upload scores.")
            return False
    else:
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
    parser.add_argument(
        "--max-job-duration",
        type=int,
        default=None,
        help="Maximum job duration in hours (default: use sbatch script default)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "slurm", "local"],
        default="auto",
        help="Deployment mode: auto (detect), slurm (use sbatch), or local (use shell script)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for local processing (ignored in slurm mode)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not upload results to Hugging Face (useful for local testing)",
    )
    parser.add_argument("--no_sanity", action="store_true", help="Skip environment sanity checks")
    parser.add_argument("--system_instruction", type=str, default=None, help="System instruction for the model")
    parser.add_argument("--tp4", action="store_true", help="Use Tensor Parallelism with 4 GPUs")
    parser.add_argument("--timestamp", action="store_true", help="Add a timestamp to the output evaluation dataset")

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Validate tasks
    tasks = [task.strip() for task in args.tasks.split(",")]
    print_info(f"Tasks to evaluate: {', '.join(tasks)}")

    # Check required environment variables
    if not check_required_env_vars(args.mode):
        sys.exit(1)

    # Activate conda environment
    if not check_conda_env(args.mode, args.watchdog):
        sys.exit(1)
    if not args.no_sanity:
        # Check required environment variables
        if not check_required_env_vars():
            sys.exit(1)

        # Activate conda environment
        if not check_conda_env(args.watchdog):
            sys.exit(1)

    # Generate timestamp and repository ID for results
    timestamp = str(int(time.time()))
    evaluation_dataset_hash = generate_evaluation_dataset_hash(tasks, args.system_instruction)
    if args.timestamp:
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

    print_header("Preparing for Distributed Evaluation Job")

    # Output directories
    output_dataset_repo_name = output_dataset.split("/")[-1]
    logs_dir = os.path.join("logs", output_dataset_repo_name)
    os.makedirs(logs_dir, exist_ok=True)
    print_info(f"Logs directory: {logs_dir}")
    output_dataset_dir = os.path.join("results", output_dataset_repo_name)
    os.makedirs(output_dataset_dir, exist_ok=True)
    print_info(f"Output dataset directory: {output_dataset_dir}")

    # Download the dataset and model
    dataset_path = download_dataset(input_dataset)
    model_path = download_model(args.model_name)

    # Determine processing mode
    if args.mode == "auto":
        # Check hostname to determine which mode to use
        cmd = "echo $HOSTNAME"
        hostname, _, _ = execute_command(cmd, verbose=False)
        if "c1" in hostname or "leonardo" in hostname:
            print_info(f"Detected cluster environment on {hostname}, using SLURM mode")
            processing_mode = "slurm"
        else:
            print_info(f"No cluster environment detected, using local mode")
            processing_mode = "local"
    else:
        processing_mode = args.mode
        print_info(f"Using user-specified {processing_mode} mode")

    # Adjust number of shards for local processing if needed
    local_num_shards = args.num_shards
    if processing_mode == "local":
        if local_num_shards > args.num_gpus:
            print_warning(
                f"Limiting number of shards to {args.num_gpus} for local processing (number of available GPUs)"
            )
            local_num_shards = args.num_gpus

    # Launch job with the dataset repo but save to output repo
    if processing_mode == "local":
        job_id = launch_local(
            model_path,
            dataset_path,
            output_dataset_dir,
            local_num_shards,
            logs_dir,
            args.tp4,
        )
    else:  # slurm mode
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
        print_info("Watchdog mode not enabled. Exiting.")
        exit(0)

    # Monitor job
    print_info("Watchdog mode enabled. Monitoring job progress...")
    monitor_job(job_id, logs_dir, args.num_shards if processing_mode == "slurm" else local_num_shards)

    # Check completion
    if not check_job_completion(job_id, output_dataset_dir):
        print_error("Some jobs failed.")
        exit(1)

    # Upload shards
    upload_shards_to_hub(output_dataset_dir, output_dataset)

    # Compute and upload scores
    if compute_and_upload_scores(tasks, output_dataset, args.model_name, logs_dir):
        print_success(f"Evaluation completed successfully. Results uploaded to {output_dataset}")
        print_info(f"View the results at: https://huggingface.co/datasets/{output_dataset}")
    else:
        print_error("Failed to compute and upload scores.")
        exit(1)


if __name__ == "__main__":
    main()
