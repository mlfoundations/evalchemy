import os
import time
from typing import Dict, List, Optional, Tuple

import ray
from lm_eval.utils import eval_logger
from ray.job_submission import JobSubmissionClient

from datasets import concatenate_datasets
from engine.dag import DAG, load_dag
from engine.executor import DAGExecutor
from engine.operators.operator import ManyShardRefs


class SyntheticDataManager:
    """
    A manager class for handling data frameworks or dataset handlers, outside of the data generation process itself.

    This class is responsible for loading, managing, and running various
    synthetic data generation frameworks based on YAML configurations.
    """

    def __init__(self):
        """
        Initialize the SyntheticDataManager.

        Sets up the strategies directory and loads all available frameworks.
        """
        self.strategies_dir = os.path.dirname(os.path.abspath(__file__))
        self.frameworks = self._load_frameworks()

    def list_frameworks(self) -> List[str]:
        """
        List all available framework names.

        Returns:
            List[str]: A list of all loaded framework names.
        """
        return list(self.frameworks.keys())

    def run_framework(self, framework_name: str, hf_account: Optional[str] = None, remote: bool = False) -> None:
        """
        Run a specific framework and push the generated dataset to Hugging Face Hub.

        Args:
            framework_name (str): The name of the framework to run.
            hf_account (str): The Hugging Face account name to push the dataset to.
            remote (bool): Whether to run the framework on a remote Ray cluster.
        Note:
            If the framework is a DatasetHandler, it will be provided with all loaded frameworks.
        """
        framework = self.frameworks[framework_name]
        if remote:
            framework.run_remote(hf_account)
            return

        if framework:
            framework.run()
            print(f"Running framework: {framework_name}")
        else:
            print(f"Framework '{framework_name}' not found.")

        if hf_account:
            framework.generated_dataset.push_to_hub(f"{hf_account}/{framework.name}")

    def _load_frameworks(self) -> Dict[str, DAG]:
        """
        Load all synthetic data frameworks from YAML configurations.

        This method scans the strategies directory for YAML files and loads
        them as either DatasetHandler or SyntheticDataFramework instances.

        Returns:
            Dict[str, SyntheticDataFramework]: A dictionary mapping framework names to their instances.

        Raises:
            ValueError: If a duplicate framework name is encountered.
        """
        frameworks = {}
        for strategy_dir in os.listdir(self.strategies_dir):
            strategy_path = os.path.join(self.strategies_dir, strategy_dir)
            if os.path.isdir(strategy_path) and strategy_dir != "__pycache__":
                for file in os.listdir(strategy_path):
                    if file.endswith(".yaml"):
                        config_path = os.path.join(strategy_path, file)
                        framework = SyntheticDataFramework.from_config(config_path)
                        frameworks[framework.name] = framework
        return frameworks


class SyntheticDataFramework:
    """
    A framework for creating and managing synthetic data generation processes.

    This class provides methods to parse a DAG (Directed Acyclic Graph) configuration,
    create operators based on the configuration, and execute ƒthe data generation process.
    """

    @staticmethod
    def from_config(config_path: str, sub_dir: Optional[Tuple[str, ...]] = None) -> "SyntheticDataFramework":
        """
        Create a SyntheticDataFramework instance from a configuration file.

        Args:
            config_path (str): Path to the configuration file.
            sub_dir (Optional[Tuple[str, ...]]): Subdirectory within the config to use.

        Returns:
            SyntheticDataFramework: An instance of the framework.
        """
        framework = SyntheticDataFramework()
        dag = load_dag(config_path, sub_dir)
        framework.executor = DAGExecutor(dag)
        framework.name = dag.name
        framework.ray_address = "http://abe691efe165e4b809c119a0bd961ae0-2123536616.us-east-1.elb.amazonaws.com:8265"
        framework.client = JobSubmissionClient(framework.ray_address)

        return framework

    def get_waitables(self) -> ManyShardRefs:
        """
        Execute the operators in the DAG and return a promise of the list of shards at the end of the data generation process.

        Returns:
            ManyShardRefs: References to the output shards of the data generation process.
        """
        return self.executor.get_waitables()

    def run(self) -> None:
        """
        Run the entire data generation process.

        This method initializes Ray, executes the DAG, and processes the results.
        """
        ray.init()
        waitables = self.get_waitables()
        ray.wait(waitables, num_returns=len(waitables))
        filtered_pairs = concatenate_datasets([ray.get(shard) for shard in waitables])
        eval_logger.info("Execution completed. Results.")
        self.generated_dataset = filtered_pairs
        ray.shutdown()

    def run_remote(self, hf_account: str) -> None:
        """
        Run the entire data generation process on a remote Ray cluster.

        This method initializes Ray, executes the DAG, and processes the results.
        """
        job_id = self.client.submit_job(
            entrypoint=f"python -m dcft.main --framework {self.name} --hf-account {hf_account}",
            runtime_env={
                "working_dir": "./",
                "pip": "./requirements.txt",
                # Exclude potentially large files and directories
                "excludes": [
                    "**/.gitignore",
                    "**/.DS_Store",
                    "**/.gitignore",
                    "**/.git",
                    "**/.venv",
                    "./datasets",
                    "**/*.json",
                    "**/*.jsonl",
                    "**/*.csv",
                ],
                "env_vars": {
                    "HF_TOKEN": os.environ["HF_TOKEN"],
                    "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                    "HF_DATASETS_CACHE": "local:///tmp/"
                },
            },
        )

        eval_logger.info(
            f"Submitted job with ID: {job_id}. Waiting for job to complete... "
            f"You can press Ctrl+C to stop and still check the status with the job ID {job_id} "
            f"at {self.ray_address}."
        )
        self._wait_until_status(job_id, ["SUCCEEDED", "FAILED"])

    def _wait_until_status(self, job_id, status_to_wait_for, timeout_seconds=36000):
        start = time.time()
        while time.time() - start <= timeout_seconds:
            status = self.client.get_job_status(job_id)
            print(f"status: {status}")
            if status in status_to_wait_for:
                break
            time.sleep(30)

        eval_logger.info(f"Job {job_id} completed with status: {status}")
