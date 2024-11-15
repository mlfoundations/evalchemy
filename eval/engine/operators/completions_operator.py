import importlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Type

import gcsfs
import ray
from bespokelabs import curator
from google.cloud import storage

from datasets import Dataset, concatenate_datasets
from engine.maps.map_registry import COMPLETIONS_MAPS
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefsGenerator,
    Operator,
    OperatorSpecificConfig,
    ShardRef,
)


class CompletionsOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for CompletionsOperator.

    Attributes:
        type (str): The type of the operator, should be 'completions'.
        materialize_output (bool): Whether to materialize the output of the operator.
        model (str): The name of the model to use for completions.
        map (str): The name of the map to use for completions.
        map_config (dict): The configuration for the map.
        batch (bool): Whether to batch the completions.
        merge_shards (bool): Whether to merge the shards of the output of the operator.
    """

    type: Literal["completions"] = "completions"
    model: str
    map: str
    map_config: dict
    batch: Optional[bool] = False
    merge_shards: Optional[bool] = True


class _DataSyncer:
    def __init__(self, local_dir: str, remote_dir: str):
        self._local_dir = local_dir
        self._remote_dir = remote_dir.replace("gs://", "")  # Remove gs:// prefix
        self._bucket_name = self._remote_dir.split("/")[0]
        self._remote_prefix = "/".join(self._remote_dir.split("/")[1:])
        self._local_mtimes = {}
        self._remote_mtimes = {}
        self._gcs_fs = gcsfs.GCSFileSystem()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _download_from_remote(self):
        """
        Perform initial sync by scanning both directories and downloading newer files
        from remote to local.
        """
        # Get remote files recursively using find
        remote_files = self._get_remote_files()
        local_files = self._get_local_files()

        all_files = remote_files.union(local_files)

        def download_single_file(path: str):
            remote_mtime = self._get_remote_mtime(path)
            local_mtime = self._get_local_mtime(path)

            # Download remote file if it's newer or local doesn't exist
            if remote_mtime and (not local_mtime or remote_mtime > local_mtime):
                local_path = os.path.join(self._local_dir, path)
                remote_path = os.path.join(self._remote_dir, path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self._gcs_fs.get(remote_path, local_path)
                local_mtime = self._get_local_mtime(path)

            if local_mtime:
                self._local_mtimes[path] = local_mtime

            if remote_mtime:
                self._remote_mtimes[path] = remote_mtime

        # Process files in parallel using ThreadPoolExecutor
        list(self._executor.map(download_single_file, all_files))

    def _get_remote_mtime(self, path: str) -> Optional[int]:
        """Get remote file modification time or None if file doesn't exist."""
        try:
            client = storage.Client()
            bucket = client.bucket(self._bucket_name)
            remote_path = os.path.join(self._remote_dir, path)
            blob_path = str(Path(remote_path).relative_to(self._bucket_name))
            blob = bucket.blob(blob_path)
            if not blob.exists():
                return None
            blob.reload()
            if blob.updated:
                return blob.updated.timestamp()
            else:
                return None
        except FileNotFoundError:
            return None

    def _get_local_mtime(self, path: str) -> Optional[float]:
        """Get local file modification time or None if file doesn't exist."""
        try:
            local_path = os.path.join(self._local_dir, path)
            return os.path.getmtime(local_path)
        except FileNotFoundError:
            return None

    def _sync_file(self, path: str) -> None:
        """Sync a single file between local and remote storage."""
        remote_mtime = self._get_remote_mtime(path)
        local_mtime = self._get_local_mtime(path)
        prev_local_mtime = self._local_mtimes.get(path, None)
        prev_remote_mtime = self._remote_mtimes.get(path, None)

        local_file_changed = (local_mtime and prev_local_mtime and local_mtime > prev_local_mtime) or (
            local_mtime and not prev_local_mtime
        )
        remote_file_changed = (remote_mtime and prev_remote_mtime and remote_mtime > prev_remote_mtime) or (
            remote_mtime and not prev_remote_mtime
        )

        if local_file_changed:
            if remote_file_changed:
                logging.warning(
                    f"Potential concurrent modification detected for file {path} since the remote file has changed without our modification. "
                    f"Local file was modified at {local_mtime} and remote file was modified at {remote_mtime}. Will not overwrite remote file."
                )
                return

            local_path = os.path.join(self._local_dir, path)
            remote_path = os.path.join(self._remote_dir, path)

            client = storage.Client()
            bucket = client.bucket(self._bucket_name)
            blob_path = str(Path(remote_path).relative_to(self._bucket_name))
            blob = bucket.blob(blob_path)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    blob.upload_from_filename(local_path)
                    self._remote_mtimes[path] = self._get_remote_mtime(path)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        logging.error(f"Failed to sync {local_path} after {max_retries} attempts: {e}")
                        raise
                    time.sleep(1)  # Exponential backoff

        self._local_mtimes[path] = self._get_local_mtime(path)
        self._remote_mtimes[path] = self._get_remote_mtime(path)

    def _scan_and_sync(self):
        """Scan both directories and sync all files."""
        remote_files = self._get_remote_files()
        local_files = self._get_local_files()

        all_files = remote_files.union(local_files)

        # Process files in parallel using ThreadPoolExecutor
        list(self._executor.map(self._sync_file, all_files))

    def _get_local_files(self):
        local_files = set()
        for root, _, files in os.walk(self._local_dir):
            for file in files:
                if file == "metadata.db":
                    continue
                full_path = os.path.join(root, file)
                local_files.add(str(Path(full_path).relative_to(self._local_dir)))
        return local_files

    def _get_remote_files(self):
        gs_paths = self._gcs_fs.find(self._remote_dir)
        return set([str(Path(path).relative_to(self._remote_dir)) for path in gs_paths if "metadata.db" not in path])


@ray.remote(num_cpus=0.1)
class _CompletionsSingleton:
    _instance = None

    def __init__(self):
        self._sync_thread = None
        self._stop_event = threading.Event()

    def _start_sync_loop(self, data_syncer: _DataSyncer, interval: int = 30):
        """Start the sync thread."""

        def sync_loop():
            while not self._stop_event.is_set():
                try:
                    data_syncer._scan_and_sync()
                except Exception as e:
                    print(f"Error during sync: {e}")
                time.sleep(interval)

            data_syncer._scan_and_sync()

        def run_sync_loop():
            sync_loop()

        self._stop_event.clear()
        self._sync_thread = threading.Thread(target=run_sync_loop, daemon=True)
        self._sync_thread.start()

    @staticmethod
    def _load_function_or_class(module_path: str) -> Callable | Type:
        try:
            module_name, attr_name = module_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            loaded_attr = getattr(module, attr_name)

            if not callable(loaded_attr):
                raise TypeError(
                    f"Loaded object '{attr_name}' from '{module_name}' is not callable. "
                    f"Expected a function or class, got {type(loaded_attr)}"
                )

            return loaded_attr
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load from '{module_path}'. "
                f"Make sure the module is in PYTHONPATH and the function/class exists. Error: {str(e)}"
            ) from e

    def completions(self, dataset: Dataset, config: CompletionsOperatorConfig, operator_id: str) -> ShardRef:
        operator_id = operator_id.replace("::", "__")
        curator_cache_dir = os.path.expanduser(f"~/.cache/curator/{operator_id}")
        os.environ["CURATOR_CACHE_DIR"] = curator_cache_dir

        remote_dir = f"dcft-data-gcp/curator-cache/{operator_id}"
        _data_syncer = _DataSyncer(curator_cache_dir, remote_dir)

        _data_syncer._download_from_remote()

        completions_map_cls = COMPLETIONS_MAPS[config.map]
        completions_map = completions_map_cls(config.map_config)
        prompt_func = completions_map.prompt
        parse_func = completions_map.parse
        response_format = completions_map.response_format

        completion = curator.Prompter(
            model_name=config.model,
            prompt_func=prompt_func,
            parse_func=parse_func,
            response_format=response_format,
            batch=config.batch,
        )

        self._start_sync_loop(data_syncer=_data_syncer)

        dataset = completion(dataset)

        self._stop_event.set()
        self._sync_thread.join(timeout=120)

        is_remote = os.environ.get("IS_REMOTE", "0") == "1"

        # This assumes the dataset is stored in a single Arrow file, which is the case
        # for the datasets we use.
        return Dataset.from_file(dataset.cache_files[0]["filename"], in_memory=is_remote)


class CompletionsOperator(Operator):
    """
    Operator for handling completions.

    Attributes:
        id (str): Unique identifier for the operator.
        input_ids (List[str]): List of input identifiers for the operator.
        config (CompletionsOperatorConfig): Specific configuration for the completions operator.
    """

    _completions_actor = None

    def __init__(self, id: str, input_ids: List[str], config: CompletionsOperatorConfig):
        super().__init__(id, input_ids, config)

    def compute(self, inputs: DatasetRefs) -> ManyShardRefsGenerator:
        """
        Compute the completions operator on the given inputs.

        Args:
            inputs (DatasetRefs): Dictionary of inputs mapping identifiers to a list of shard references.

        Returns:
            ManyShardRefsGenerator: Generator of processed output shard references for each input shard.
        """
        if self.config.merge_shards:
            shard_refs = [shard_ref for (_, shard_refs) in inputs.items() for shard_ref in shard_refs]
            waitable = self.merge_shards.remote(shard_refs)
            yield self._get_completions_actor().completions.remote(waitable, self.config, self._id)
            return

        for _, shard_refs in inputs.items():
            for shard_ref in shard_refs:
                waitable = self._get_completions_actor().completions.remote(shard_ref, self.config, self._id)
                yield waitable

    @staticmethod
    @ray.remote
    def merge_shards(shard_refs: List[ShardRef]) -> ShardRef:
        dataset_shards = []
        for shard_ref in shard_refs:
            dataset_shards.append(ray.get(shard_ref))
        return concatenate_datasets(dataset_shards)

    def _get_completions_actor(self):
        if self._completions_actor is None:
            self._completions_actor = _CompletionsSingleton.options(name="CompletionsActor").remote()
        return self._completions_actor

    def cleanup(self):
        """Clean up resources when the operator is being shut down."""
        ray.get(self._completions_actor.shutdown.remote())
