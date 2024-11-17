import datetime
import json
import logging
import os
import subprocess
import time
import uuid
from itertools import tee
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterator, List, Optional

import fsspec
import gcsfs
import huggingface_hub
import psycopg
import ray
from lm_eval.utils import eval_logger
from psycopg.types.json import Jsonb
from pydantic import ValidationError
from ray.job_submission import JobSubmissionClient

from datasets import (
    Dataset,
    concatenate_datasets,
    disable_caching,
    load_dataset,
    load_from_disk,
)
from dcft.data_strategies.yaml_utils import remove_prefix, walk_directory
from engine.dag import DAG
from engine.operators.function_operator import FunctionOperatorConfig
from engine.operators.hashing_utils import HashCodeHelper
from engine.operators.operator import (
    ManyShardRefsGenerator,
    OperatorConfig,
    ShardRef,
    create_operator,
    parse_specific_config,
    parse_yaml_config,
)

# Disable caching for remote runs. Otherwise, datasets will be unserializable for Ray
# and cause bugs when they're passed as refs.
if os.environ.get("IS_REMOTE", "0") == "1":
    disable_caching()


def flatten(nested_list: List[Any]) -> List[Any]:
    """
    Flatten a nested list structure.

    Args:
        nested_list (List[Any]): A list that may contain nested lists.

    Returns:
        List[Any]: A flattened version of the input list.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except subprocess.CalledProcessError:
        return None


def get_git_diff():
    try:
        staged_diff = subprocess.check_output(["git", "diff", "--staged"]).decode("utf-8")
        unstaged_diff = subprocess.check_output(["git", "diff"]).decode("utf-8")
        diff = f"===Staged changes===\n{staged_diff}\n===Unstaged changes===\n{unstaged_diff}"
    except subprocess.CalledProcessError:
        return None
    return diff


def _to_generator(iter: Iterator[ShardRef]) -> ManyShardRefsGenerator:
    """
    Convert an iterator to a generator.
    """
    yield from iter


class SyntheticDataManager:
    """
    A manager class for handling data frameworks or dataset handlers, outside of the data generation process itself.

    This class is responsible for loading, managing, and running various
    synthetic data generation frameworks based on YAML configurations.
    """

    def __init__(
        self,
        hf_account: str,
        output_dir: Optional[str] = None,
        fs_type: str = "file",
        hf_private: bool = True,
        ray_address: Optional[str] = None,
        no_return: bool = False,
        max_pending_waitables: int = 100,
        db_connection_string: Optional[str] = None,
        enable_cache: bool = False,
        log_level: str = "INFO",
        resume_from_partial: bool = False,
    ):
        """
        Initialize the SyntheticDataManager.

        Args:
            hf_account (str): The Hugging Face account name.
            fs_type (str, optional): The filesystem type to use. Defaults to "file".
            ray_address (Optional[str], optional): The address of the Ray cluster. Defaults to None.
            no_return (bool, optional): Whether to not return data to the local machine. Defaults to False.
            max_pending_waitables (int, optional): The maximum number of waitables to wait on at once. Defaults to 100.
            db_connection_string (Optional[str], optional): The connection string for the PostgreSQL database. Defaults to None.
            enable_cache (bool, optional): Whether to enable caching. Defaults to False.
            log_level (str, optional): The log level to use. Defaults to "INFO".
            resume_from_partial (bool, optional): Whether to use existing shards in a partially completed operator's cache even if not fully finished
        """
        self.hf_account = hf_account
        self.hf_private = hf_private
        self.no_return = no_return
        self.fs_type = fs_type
        if fs_type != "gcs":
            self.fs = fsspec.filesystem(fs_type)
        else:
            self.fs = gcsfs.GCSFileSystem(project="bespokelabs")
        self.strategies_dir = os.path.dirname(os.path.abspath(__file__))
        self.frameworks = self._load_frameworks()
        self.ray_address = ray_address
        self.max_pending_waitables = max_pending_waitables
        self.output_dir = output_dir
        self.enable_cache = enable_cache

        self.db_connection_string = db_connection_string
        if self.db_connection_string:
            self.db_connection = psycopg.connect(self.db_connection_string)
        else:
            self.db_connection = None
        self.created_dataset_ids = []
        self.log_level = log_level
        self.resume_from_partial = resume_from_partial

    def list_frameworks(self) -> List[str]:
        """
        List all available framework names.

        Returns:
            List[str]: A list of all loaded framework names.
        """
        return list(self.frameworks.keys())

    def run_framework(self, framework_name: str, remote: bool = False) -> None:
        """
        Run a specific framework and push the generated dataset to Hugging Face Hub.

        Args:
            framework_name (str): The name of the framework to run.
            remote (bool, optional): Whether to run the framework on a remote Ray cluster. Defaults to False.
        """
        log_level = getattr(logging, self.log_level)

        framework_path = self.frameworks.get(framework_name, None)
        self.framework_name = framework_name
        self.parsed_yamls = set()
        if framework_path is None:
            raise ValueError(f"Framework '{framework_name}' not found.")

        eval_logger.info(f"Running framework: {framework_name}")

        if remote:
            self.run_remote(framework_name)
        else:

            def logging_setup_func():
                logging.basicConfig(level=log_level)

            ray.init(logging_level=log_level, runtime_env={"worker_process_setup_hook": logging_setup_func})
            logging_setup_func()
            self.from_config(framework_path)
            try:
                self.run()
            finally:
                ray.shutdown()

        if self.hf_account and not self.no_return:
            commit_message = f"Automatic dcft datacuration framework upload for {framework_name}"
            repo_id = f"{self.hf_account}/{framework_name}"

            # Upload the generated dataset to the Hugging Face Hub
            self.generated_dataset.push_to_hub(repo_id=repo_id, commit_message=commit_message, private=self.hf_private)

            # Upload the data generation configuration file to the Hugging Face Hub
            for framework_path in self.parsed_yamls:
                huggingface_hub.upload_file(
                    path_or_fileobj=framework_path,
                    path_in_repo=os.path.join("config", remove_prefix(framework_path)),
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=commit_message,
                )

            uuid_string = str(uuid.uuid4())
            uploaded_dataset = load_dataset(repo_id)

            # when --dev is enabled, --db_connection_string is set to "" and self.db_connection is None, so we don't save to the database
            if self.db_connection:
                self._save_dataset_info_to_db(
                    uuid_string,
                    repo_id,
                    self.dag.to_dict(),
                    datetime.datetime.now(datetime.timezone.utc),
                    hf_fingerprint=uploaded_dataset["train"]._fingerprint,
                    external_link=f"https://huggingface.co/datasets/{repo_id}",
                )
            eval_logger.info(
                f"Uploaded dataset and config to https://huggingface.co/datasets/{repo_id} with uuid {uuid_string}"
            )
        self.cleanup()

    def _load_frameworks(self) -> Dict[str, str]:
        """
        Load all synthetic data frameworks from YAML configurations.
        Uses utility function to recursively search through all subdirectories.

        Returns:
            Dict[str, str]: A dictionary mapping framework names to their config file paths.

        Raises:
            ValueError: If a duplicate framework name is encountered.
        """
        return walk_directory(
            directory=self.strategies_dir, file_extensions=(".yaml", ".yml"), skip_dirs=("__pycache__",)
        )

    def from_config(self, config_path: str) -> None:
        """
        Create a DAG from a configuration file.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.dag = self.parse_dag(config_path)

    def get_waitables(self) -> List[ManyShardRefsGenerator]:
        """
        Execute the operators in the DAG and return a list of waitables representing the output shards.

        Returns:
            List[ManyShardRefsGenerator]: References to the output shards of the data generation process.
        """
        datas: Dict[str, ManyShardRefsGenerator] = {}

        sorted_ops = self.dag.topological_sort()
        out_degree_map = self.dag.get_out_degree_map()
        waitables = []

        hasher = HashCodeHelper()
        self.map_op_id_to_dag_hash = self.dag.calculate_operator_hashes(sorted_ops, hasher)

        dag_dict = self.dag.to_dict()
        waitables = []
        self.logging_waitables = {self.map_op_id_to_dag_hash[op.id]: [] for op in sorted_ops}

        print(f"out_degree_map: {out_degree_map}")

        for operator in sorted_ops:
            # Prepare input data for the operator
            input_datas = {}
            for input_id in operator.input_ids:
                if out_degree_map[input_id] > 1:
                    # Since the input_ids is still needed by more than one operator, we need to
                    # tee the generator so that the output operators can independently consume the shards.
                    iter1, iter2 = tee(datas[input_id])
                    input_datas[input_id] = _to_generator(iter1)
                    datas[input_id] = _to_generator(iter2)
                else:
                    input_datas[input_id] = datas[input_id]

                # Decrement the out-degree of the input operator
                out_degree_map[input_id] -= 1

            print(f"Operator {operator.id} has input_datas:{list(input_datas.keys())}")
            # Execute the operator, load from cache if possible
            loaded_from_fs = False
            if (
                self.output_dir
                and self.fs
                and self.enable_cache
                and self.fs.exists(f"{self.output_dir}/{self.map_op_id_to_dag_hash[operator.id]}")
            ):
                if self.resume_from_partial or self.fs.exists(
                    f"{self.output_dir}/{self.map_op_id_to_dag_hash[operator.id]}/SUCCESS_FLAG"
                ):
                    logging.info(f">>> Found {self.map_op_id_to_dag_hash[operator.id]} for operator {operator.id}")
                    if not self.fs.exists(f"{self.output_dir}/{self.map_op_id_to_dag_hash[operator.id]}/SUCCESS_FLAG"):
                        logging.info(
                            f"Resuming from partial at {self.output_dir}/{self.map_op_id_to_dag_hash[operator.id]}"
                        )
                    curr_op_output = self._load_dataset_from_fs.options(name=f"load_from_cache_{operator.id}").remote(
                        self.output_dir, self.map_op_id_to_dag_hash[operator.id], self.fs
                    )
                    loaded_from_fs = True
                else:
                    eval_logger.warning(
                        f"Partial dataset exists at {self.output_dir}/{self.map_op_id_to_dag_hash[operator.id]}, overwriting the cache."
                    )

            if not loaded_from_fs:
                curr_op_output = operator.execute(input_datas)

            if operator.config.materialize_output and not loaded_from_fs and operator.config.type != "hf_source":
                generation_parameters = dag_dict.copy()
                generation_parameters["op_id"] = operator.id
                dataset_id = str(uuid.uuid4())
                generation_start = datetime.datetime.now(datetime.timezone.utc)

                if self.db_connection:
                    self._save_dataset_info_to_db(dataset_id, operator.id, generation_parameters, generation_start)

                curr_op_output = self._wrap_generator_with_logging(
                    curr_op_output, dataset_id, operator.id, self.map_op_id_to_dag_hash[operator.id]
                )
            if operator.id in self.dag.output_ids:
                waitables.append(curr_op_output)

            datas[operator.id] = curr_op_output

        eval_logger.info(f"Generated {len(waitables)} waitables")

        return waitables

    def wait_for_results(self, waitables: List[ManyShardRefsGenerator], no_return: bool = False) -> List[Dataset]:
        """
        Wait for the waitables and return the results.

        The goal is to generate the data in a way that doesn't overwhelm the
        system's distributed memory. We do this by controlling how many waitables
        are in-flight at the same time (if the number of waitables is too high, we
        wait for some of them to finish before adding more). Note that this is
        only possible if the waitables are generators and we can control when
        new shards are generated.

        Args:
            waitables (List[ManyShardRefsGenerator]): List of waitables to process.
            no_return (bool, optional): Whether to not return data to the local machine. Defaults to False.

        Returns:
            List[Dataset]: The results obtained from the waitables as a list of Dataset objects.
        """
        i = 0
        pending_waitables = []
        results = []

        while (
            i < len(waitables)
            or len(pending_waitables) > 0
            or self.dataset_end_waitables
            or remaining_logging_waitables
        ):
            if i < len(waitables):
                try:
                    shard = next(waitables[i])
                    pending_waitables.append(shard)
                except StopIteration:
                    i += 1

            # Some waitables are actually done, but if we don't ray.wait them and garbage collect their
            # references, they'll keep accumulating in the object store. In order to avoid this, we
            # periodically ray.wait on the pending_waitables.
            if len(pending_waitables) > self.max_pending_waitables or i >= len(waitables):
                ready_waitables, pending_waitables = ray.wait(pending_waitables, fetch_local=False, timeout=30)
                for ready_waitable in ready_waitables:
                    if not no_return:
                        dataset = ray.get(ready_waitable)
                        results.append(dataset)
                eval_logger.info(f"Finished waiting. Remaining: {len(pending_waitables)} waitables")

            # Update end times for datasets that have finished
            if self.dataset_end_waitables:
                for dataset_end_waitable in self.dataset_end_waitables:
                    ready, _ = ray.wait([dataset_end_waitable.last_item], fetch_local=False, timeout=0.1)
                    if ready:
                        self._update_dataset_end_times(dataset_end_waitable.dataset_id)
                        self.dataset_end_waitables.remove(dataset_end_waitable)

            remaining_logging_waitables = True
            if remaining_logging_waitables:
                remaining_logging_waitables = False
                for operator_hash in self.logging_waitables:
                    lst = self.logging_waitables[operator_hash]
                    ready_waitable, pending_logging_waitables = ray.wait(lst, fetch_local=False, timeout=30)
                    if pending_logging_waitables:
                        self.logging_waitables[operator_hash] = pending_logging_waitables
                        remaining_logging_waitables = True
                        eval_logger.info(f"Still saving shards at {self.output_dir}/{operator_hash}")
                    else:
                        self.fs.makedirs(f"{self.output_dir}/{operator_hash}/SUCCESS_FLAG", exist_ok=True)
                        eval_logger.info(f"Success flag planted at {self.output_dir}/{operator_hash}")
        eval_logger.info(f"Finished waiting for waitables")
        return results

    def run(self) -> None:
        """
        Run the entire data generation process.

        This method executes the DAG, processes the results, and stores the generated dataset.
        """
        self.run_id = str(uuid.uuid4())
        eval_logger.info(f"Run ID: {self.run_id}")

        self.dataset_end_waitables = []
        self._initialize_git_info()

        waitables = self.get_waitables()
        results = self.wait_for_results(waitables, no_return=self.no_return)
        if self.no_return:
            eval_logger.info("Execution completed. Not returning results.")
        else:
            filtered_pairs = concatenate_datasets(results)
            eval_logger.info("Execution completed.")
            self.generated_dataset = filtered_pairs
            if self.output_dir:
                self.generated_dataset.to_json(f"{self.output_dir}/generated.jsonl")

    def run_remote(self, framework_name: str) -> None:
        """
        Run the entire data generation process on a remote Ray cluster.

        Args:
            framework_name (str): The name of the framework to run.
        """
        self.client = JobSubmissionClient(self.ray_address)
        cmd_args = [
            f"--framework {framework_name}",
        ]

        if self.hf_account:
            cmd_args.append(f"--hf-account {self.hf_account}")
        if self.no_return:
            cmd_args.append("--no-return")
        if self.fs_type:
            cmd_args.append(f"--fs-type {self.fs_type}")
        if self.enable_cache:
            cmd_args.append("--enable-cache")
        if self.max_pending_waitables:
            cmd_args.append(f"--max-pending-waitables {self.max_pending_waitables}")
        if self.output_dir:
            cmd_args.append(f"--output-dir {self.output_dir}")
        if self.db_connection_string:
            cmd_args.append(f"--db-connection-string {self.db_connection_string}")

        requirements = []
        with open("requirements.txt", "r") as f:
            for line in f:
                if line.startswith("."):
                    # Replace the relative path with the absolute path based
                    # on the working directory for the Ray worker.
                    # (see https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-local-files)
                    line = line.replace("./", "${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/")
                    requirements.append(line)
                    continue
                if line.strip() and not line.startswith("#"):
                    requirements.append(line.strip())

        job_id = self.client.submit_job(
            entrypoint=f"python -m dcft.main {' '.join(cmd_args)}",
            runtime_env={
                "working_dir": "./",
                "pip": {"packages": requirements},
                "py_modules": ["dcft/external_repositories/MetaMath/dist/MetaMathAgain-0.2.0-py3-none-any.whl"],
                # Exclude potentially large files and directories
                "excludes": [
                    "**/.gitignore",
                    "**/.DS_Store",
                    "**/.gitignore",
                    "**/.git",
                    "**/.venv",
                    "/datasets",
                    "/eval",
                    "**/*.csv",
                    "**/*.bin",
                    "**/*.jsonl",
                    "**/*.gif",
                ],
                "env_vars": {
                    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
                    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                    "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", ""),
                    "RAY_DEDUP_LOGS": "0",
                    "RAY_TASK_MAX_RETRIES": "-1",
                    "SYNTHETIC_DATA_MANAGER_CREATION_LOCATION": self.ray_address,
                    "GIT_COMMIT_HASH": get_git_commit_hash(),
                    "GIT_DIFF": get_git_diff(),
                    "IS_REMOTE": "1",
                },
            },
        )

        eval_logger.info(
            f"Submitted job with ID: {job_id}. Waiting for job to complete... "
            f"You can press Ctrl+C to stop and still check the status with the job ID {job_id} "
            f"at {self.ray_address}."
        )
        self._wait_until_status(job_id, ["SUCCEEDED", "FAILED"])

    def cleanup(self) -> None:
        """
        Clean up and save the generated datasets to cache.
        """
        if self.db_connection:
            self.db_connection.close()

    def _look_up_dataset_id(self, hash_id: str) -> str:
        if not self.db_connection:
            return None

        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id FROM datasets
                    WHERE data_generation_hash = %s
                    AND generation_status = 'COMPLETED'
                    ORDER BY generation_end DESC
                    LIMIT 1
                    """,
                    (hash_id,),
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            eval_logger.error(f"Error looking up dataset ID: {e}")
            return None

    def _save_dataset_info(
        self, dataset_id: str, name: str, generation_parameters: dict, op_id: str, generation_start: datetime.datetime
    ):
        if self.db_connection:
            self._save_dataset_info_to_db(dataset_id, op_id, generation_parameters, generation_start)

    def _save_dataset_info_to_db(
        self,
        dataset_id: str,
        name: str,
        generation_parameters: dict,
        generation_start: datetime.datetime,
        hf_fingerprint: Optional[str] = "",
        external_link: Optional[str] = "",
    ):
        if not self.db_connection:
            eval_logger.warning("Database connection not available. Skipping database save.")
            return

        try:
            data_generation_hash = None
            if self.map_op_id_to_dag_hash and (name in self.map_op_id_to_dag_hash):
                data_generation_hash = self.map_op_id_to_dag_hash[name]

            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO datasets (
                        id, run_id, name, created_by, creation_location, creation_time, 
                        generation_start, data_location, generation_parameters, 
                        generation_status, dataset_type, is_external, is_sharded,
                        data_generation_hash, git_commit_hash, git_diff, hf_fingerprint, external_link
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        dataset_id,
                        self.run_id,
                        name,
                        "SyntheticDataManager",
                        os.environ.get("SYNTHETIC_DATA_MANAGER_CREATION_LOCATION", "unknown"),
                        datetime.datetime.now(datetime.timezone.utc),
                        generation_start,
                        f"{self.output_dir}/{dataset_id}" if self.output_dir and self.fs_type != "file" else None,
                        Jsonb(generation_parameters),
                        "STARTED",
                        "SFT",
                        False,
                        True,
                        data_generation_hash,
                        self.git_commit_hash,
                        self.git_diff,
                        hf_fingerprint,
                        external_link,
                    ),
                )
            self.db_connection.commit()
            self.created_dataset_ids.append(dataset_id)  # Add the dataset ID to our list
            eval_logger.info(f"Saved dataset info to database for dataset_id: {dataset_id}")
        except Exception as e:
            self.db_connection.rollback()
            eval_logger.error(f"Error saving to database: {e}")

    def _update_dataset_end_times(self, dataset_id: str):
        if not self.db_connection:
            return

        try:
            with self.db_connection.cursor() as cursor:
                generation_end = datetime.datetime.now(datetime.timezone.utc)
                cursor.execute(
                    """
                    UPDATE datasets
                    SET generation_end = %s, generation_status = 'COMPLETED'
                    WHERE id = %s
                """,
                    (generation_end, dataset_id),
                )
            self.db_connection.commit()
            eval_logger.info(f"Updated end times for dataset {dataset_id}")
        except Exception as e:
            self.db_connection.rollback()
            eval_logger.error(f"Error updating dataset end times: {e}")

    def _wrap_generator_with_logging(self, generator, dataset_id, operator_id, operator_hash):
        for i, item in enumerate(generator):
            if self.output_dir:
                self.logging_waitables[operator_hash].append(
                    self._save_shard.remote(
                        item,
                        i,
                        self.output_dir,
                        dataset_id,
                        self.fs.open,
                        self.fs_type,
                        operator_id,
                        operator_hash,
                        self.fs,
                    )
                )
            yield item

    @staticmethod
    @ray.remote
    def _save_shard(dataset, idx, output_dir, dataset_id, custom_open, fs_type, operator_id, operator_hash, fs):
        operator_id = operator_id.replace("::", "/")
        os.makedirs(f"{output_dir}/{operator_hash}/", exist_ok=True)
        logging.info(f"Saving shard {idx} to {output_dir}/{operator_hash}/{idx}")
        dataset.save_to_disk(f"{output_dir}/{operator_hash}/{idx}", storage_options={"open": custom_open})
        if fs.exists(f"{output_dir}/shard_info.json"):
            with fs.open(f"{output_dir}/shard_info.json", "r") as f:
                data = json.load(f)
                logging.info(f"Successfully loaded data from {output_dir}/shard_info.json")
        else:
            data = {}
            logging.info(f"File {output_dir}/shard_info.json not found. Starting with empty list.")

        data[f"{dataset_id}/{operator_id}"] = operator_hash

        # Save updated data back to file
        with fs.open(f"{output_dir}/shard_info.json", "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    @ray.remote
    def _load_dataset_from_fs(output_dir, operator_hash, fs):
        path = f"{output_dir}/{operator_hash}"

        # List all files/directories in the path
        contents = fs.listdir(path)

        # Extract shard indices and sort
        shard_paths = []
        for item in contents:
            if item["name"].endswith("info.json") or item["name"].endswith("SUCCESS_FLAG"):
                continue
            full_path = item["name"]
            shard_idx = int(full_path.split("/")[-1])
            shard_paths.append((shard_idx, full_path))
        # Sort by shard index
        shard_paths.sort(key=lambda x: x[0])

        # Load and yield datasets in sorted order
        for _, dataset_path in shard_paths:
            logging.warning(f"Attempt to load from {dataset_path}")
            dataset = load_from_disk(
                dataset_path,
                storage_options={"open": fs.open},
                keep_in_memory=(os.environ.get("IS_REMOTE", "0") == "1"),
            )
            yield dataset

    def _wait_until_status(self, job_id: str, status_to_wait_for: List[str], timeout_seconds: int = 36000) -> None:
        """
        Wait until the job reaches a specified status.

        Args:
            job_id (str): The ID of the job to wait for.
            status_to_wait_for (List[str]): List of statuses to wait for.
            timeout_seconds (int, optional): Timeout in seconds. Defaults to 36000.
        """
        start = time.time()
        while time.time() - start <= timeout_seconds:
            status = self.client.get_job_status(job_id)
            eval_logger.info(f"status: {status}")
            if status in status_to_wait_for:
                break
            time.sleep(30)

        eval_logger.info(f"Job {job_id} completed with status: {status}")

    def parse_dag(self, config_path: str) -> DAG:
        """
        Parse the configuration and create a DAG.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            DAG: The created DAG.

        Raises:
            ValueError: If there are duplicate operator IDs, invalid configurations, or invalid DAG structure.
        """
        dag = DAG()

        seen_ids = set()
        self.parsed_yamls.add(config_path)
        queue_config_paths: Queue = Queue()
        queue_config_paths.put((None, config_path))

        renaming_map: Dict[str, List[str]] = {}
        config = parse_yaml_config(config_path)
        config["name"] = Path(config_path).stem
        renaming_map = {}
        for op_config in config["operators"]:
            op_id = f"{config['name']}::{op_config['id']}"
            if op_id in seen_ids:
                raise ValueError(f"Duplicate operator ID found: {op_id}")
            seen_ids.add(op_id)

            if op_config["config"]["type"] == "load_preexisting":
                sub_dag = self.parse_dag(self.frameworks[op_config["config"]["framework_name"]])
                dag.extend(sub_dag)
                renaming_map[op_id] = sub_dag.output_ids
            else:
                try:
                    specific_config = parse_specific_config(op_config["config"])
                    if "input_ids" in op_config:
                        inpid = [f"{config['name']}::{input_id}" for input_id in op_config["input_ids"]]
                    else:
                        inpid = []

                    if isinstance(specific_config, FunctionOperatorConfig):
                        if len(specific_config.input_dataset_map.keys()) > 0:
                            for key, value in specific_config.input_dataset_map.items():
                                specific_config.input_dataset_map[key] = f"{config['name']}::{value}"

                    operator_config = OperatorConfig(id=op_id, input_ids=inpid, config=specific_config)

                    operator = create_operator(operator_config)
                    dag.add_operator(operator)

                except ValidationError as e:
                    raise ValueError(f"Invalid configuration for operator {op_id}: {str(e)}")

        # If output_ids is not specified, use the last operator's ID
        if "output_ids" not in config:
            if dag.operators:
                output_of_sub_dag = [dag.operators[-1].id]
        else:
            output_of_sub_dag = [f"{config['name']}::{item}" for item in config["output_ids"]]

        dag.set_output_ids(output_of_sub_dag)

        for operator in dag.operators:
            operator.set_input_ids(flatten([renaming_map.get(item, item) for item in operator.input_ids]))

        try:
            dag.validate()
        except ValueError as e:
            raise ValueError(f"Invalid DAG structure: {str(e)}")

        dag.operators = dag.topological_sort()
        return dag

    def _initialize_git_info(self):
        if os.environ.get("GIT_COMMIT_HASH"):
            self.git_commit_hash = os.environ.get("GIT_COMMIT_HASH")
        else:
            self.git_commit_hash = get_git_commit_hash()

        if os.environ.get("GIT_DIFF"):
            self.git_diff = os.environ.get("GIT_DIFF")
        else:
            self.git_diff = get_git_diff()
