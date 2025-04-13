#!/usr/bin/env python3
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

from database.models import EvalResult, Model
from database.utils import get_model_from_db, get_or_add_model_by_name, session_scope
from eval.eval_tracker import DCEvaluationTracker

# Configuration
TASKS = [
    # "MTBench",
    # "MixEval",
    # "HumanEval",
    # "IFEval",
    # "mmlu",
    # "arc_challenge",
    # "drop",
    # "MBPP",
    # "WildBench",
    # "hendrycks_math",
    # "gsm8k",
    "AIME24",
    "AMC23",
    # "BigCodeBench",
    # "scibench",
    "MATH500",
    "GPQADiamond",
    "LiveCodeBench",
    "LiveBench",
]
BASE_PORT = 29500
SLURM_DIR = "slurm_scripts"
LOG_DIR = "log/slurm"
CHECK_INTERVAL = 600  # 10 minutes between database checks
MAX_CONCURRENT_JOBS = 10
CACHE_DIR = "/tmp/hf_home/"  # Match HF_HOME from original script


class AutoEvalManager:
    def __init__(self):
        self.setup_directories()
        self.tracker = DCEvaluationTracker("logs", use_database=True)
        self.completed_tasks = {}  # type: Dict[str, Set[str]]
        self.submitted_jobs = set()  # type: Set[tuple]
        self.active_models = dict()  # type: Dict[str, str]  # UUID -> HF model path

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        Path(SLURM_DIR).mkdir(parents=True, exist_ok=True)
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    def check_for_new_models(self) -> Dict[str, str]:
        """Query database for new models since start of 2025."""
        new_models = {}
        with session_scope() as session:
            # Get all completed models from 2025 onwards
            query = (
                session.query(Model)
                .filter(
                    Model.training_status == "Done",
                    Model.last_modified >= datetime(2025, 1, 22, tzinfo=timezone.utc),
                )
                .all()
            )
            for model in query:
                model_id = str(model.id)
                if model_id not in (self.active_models or {}) and model_id not in (self.completed_tasks or {}):
                    new_models[model_id] = model.weights_location.replace("hf://", "")
        return new_models

    # def download_model(self, model_id: str, repo_id: str) -> bool:
    #     """Download model files to cache without loading."""
    #     try:
    #         # Download to cache
    #         print("downloading model to ", f"CACHE_DIR: {CACHE_DIR}")
    #         snapshot_download(repo_id=repo_id, cache_dir=CACHE_DIR, local_files_only=False)
    #         return True
    #     except Exception as e:
    #         print(f"Failed to download model {model_id} ({repo_id}): {e}")
    #         return False
    def check_model_completion(self, model_id: str) -> bool:
        """Check if all tasks are completed for a model."""
        if model_id not in self.completed_tasks:
            self.completed_tasks[model_id] = set()
        completed_count = len(self.completed_tasks[model_id])
        return completed_count == len(TASKS)

    def get_task_status(self, uuid: str, task: str) -> bool:
        """Check if a task is completed."""
        if uuid not in self.completed_tasks:
            self.completed_tasks[uuid] = set()
        if task not in self.completed_tasks[uuid]:
            if self.tracker.check_if_already_done(task, uuid):
                self.completed_tasks[uuid].add(task)
        return task in self.completed_tasks[uuid]
        completed_count = len(self.completed_tasks[model_id])
        return completed_count == len(TASKS)

    def generate_slurm_script(self, uuid: str, port: int) -> str:
        """Generate SLURM script content."""
        script = f"""#!/bin/bash
#SBATCH --job-name=run_{uuid}
#SBATCH --mail-type=FAIL,INVALID_DEPEND
#SBATCH --mail-user=neginmr@utexas.edu
#SBATCH --partition=ckpt-all
#SBATCH --nodes={len(TASKS)}
#SBATCH -p gh
#SBATCH --cpus-per-task=40
#SBATCH --account CCR24067
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/08134/negin/ls6/evalchemy
#SBATCH --export=all
#SBATCH --output=slurm_logs/%j_{uuid}.out
#SBATCH --error=slurm_logs/%j_{uuid}.err
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
module load cuda/12.4 nccl/12.4
cd /work/08134/negin/ls6/evalchemy
source /work/08134/negin/anaconda3/etc/profile.d/conda.sh
conda activate evalchemy
export HF_TOKEN=<HF TOKEN HERE>
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
export OPENAI_API_KEY=<OPENAI API KEY HERE>
export HF_HOME={CACHE_DIR}
export MASTER_PORT={port}
"""
        for i, task in enumerate(TASKS):
            script = (
                script
                + f"""srun  --nodes=1 --ntasks=1 -w "${{nodes_array[{i}]}}" python -m eval.eval \\
    --model vllm --tasks {task} \\
    --annotator_model gpt-4o-mini-2024-07-18 \\
    --model_id {uuid} \\
    --batch_size "auto" \\
    --output_path logs \\
    --use_database &
    """
            )
        # Wait for all to complete
        script = script + "wait"
        return script

    def submit_job(self, uuid: str) -> bool:
        """Submit a job if not already submitted."""
        job_key = uuid
        if job_key in self.submitted_jobs:
            return False
        port = BASE_PORT + random.randint(0, 999)
        print("port-", port)
        script_content = self.generate_slurm_script(uuid, port)
        script_path = f"{SLURM_DIR}/{uuid}.sh"
        try:
            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            subprocess.run(["sbatch", script_path], check=True)
            self.submitted_jobs.add(job_key)
            return True
        except Exception as e:
            print(f"Failed to submit job for model {uuid}: {e}")
            return False

    def check_running_jobs(self) -> int:
        """Check SLURM queue and update submitted_jobs."""
        try:
            result = subprocess.run(
                ["squeue", "-h", "-u", os.getenv("USER"), "-O", "Name:60"], capture_output=True, text=True
            )
            running_jobs = len(result.stdout.splitlines())
            if running_jobs < len(self.submitted_jobs):
                for line in result.stdout.splitlines():
                    if "run_" in line:
                        parts = line.split()
                        job_name = parts[3]
                        _, uuid = job_name.split("_", 1)
                        self.submitted_jobs.add(uuid)
            return running_jobs
        except Exception as e:
            print(f"Error checking running jobs: {e}")
            return len(self.submitted_jobs)

    def cleanup_model_cache(self, model_id: str) -> None:
        """Remove model files from HF cache after completion."""
        try:
            repo_id = self.active_models[model_id]
            cache_path = os.path.join(CACHE_DIR, "models--" + repo_id.replace("/", "--"))
            if os.path.exists(cache_path):
                print(f"\nCleaning up cache for completed model {model_id} at {cache_path}")
                shutil.rmtree(cache_path)
                print(f"Successfully removed cache for model {model_id}")
            else:
                print(f"No cache found for model {model_id} at {cache_path}")
        except Exception as e:
            print(f"Error cleaning up cache for model {model_id}: {e}")

    def run(self):
        """Main loop to monitor for new models and manage evaluations."""
        print("Starting continuous model monitoring and evaluation...")
        while True:
            # Check for new models
            new_models = self.check_for_new_models()
            if new_models:
                print(f"\nFound {len(new_models)} new models!")
                for model_id, repo_id in new_models.items():
                    print(f"Downloading model {model_id} ({repo_id})...")
                    self.active_models[model_id] = repo_id
                    # if self.download_model(model_id, repo_id):
                    #     self.active_models[model_id] = repo_id
                    #     print(f"Successfully downloaded model {model_id}")
                    # else:
                    #     print(f"Failed to download model {model_id}")
            # Remove completed models
            completed_models = [
                model_id for model_id in list(self.active_models.keys()) if self.check_model_completion(model_id)
            ]
            for model_id in completed_models:
                print(f"\nModel {model_id} has completed all tasks. Cleaning up...")
                # self.cleanup_model_cache(model_id)
                del self.active_models[model_id]
            # Check running jobs and available slots
            running_jobs = self.check_running_jobs()
            available_slots = MAX_CONCURRENT_JOBS - running_jobs
            if available_slots <= 0:
                print(f"\nMaximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue
            # Submit new jobs for active models
            for model_id in tqdm(self.active_models):
                if model_id in self.submitted_jobs:
                    continue
                if available_slots <= 0:
                    break
                if self.submit_job(model_id):
                    print(f"Submitted new job for model {model_id}")
                    available_slots -= 1
            # Print status update
            print("\nCurrent Status:")
            print(f"Active models: {len(self.active_models)}")
            print(f"Running jobs: {running_jobs}")
            print(f"Submitted jobs: {len(self.submitted_jobs)}")
            print(f"\nWaiting {CHECK_INTERVAL/60} minutes before next check...")
            time.sleep(CHECK_INTERVAL)


def main():
    manager = AutoEvalManager()
    manager.run()


if __name__ == "__main__":
    main()
