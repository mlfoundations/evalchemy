import fcntl
import hashlib
import logging
import os
import subprocess
import tempfile
import time
from typing import Literal, Optional

import fasttext
import numpy as np
import ray
import requests
from pydantic import DirectoryPath, Field, HttpUrl

from datasets import Dataset
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefsGenerator,
    Operator,
    OperatorSpecificConfig,
)

logger = logging.getLogger(__name__)


class FastTextOperatorConfig(OperatorSpecificConfig):
    type: Literal["fasttext"] = "fasttext"
    fasttext_url: str
    input_column: str
    filter_threshold: float = Field(default=0.5, ge=0, le=1)
    cache_dir: Optional[DirectoryPath] = None
    batch_size: int = Field(default=32, ge=1)

    class Config:
        extra = "forbid"


class FastTextOperator(Operator):
    def __init__(self, id: str, input_ids: list[str], config: FastTextOperatorConfig):
        super().__init__(id, input_ids, config)
        self.fasttext_url = str(config.fasttext_url)
        self.input_column = config.input_column
        self.filter_threshold = config.filter_threshold
        self.cache_dir = (
            os.path.join(os.getcwd(), ".cache", "fasttext") if config.cache_dir is None else config.cache_dir
        )
        self.batch_size = config.batch_size

    def compute(self, inputs: DatasetRefs) -> ManyShardRefsGenerator:
        for input in inputs.values():
            for shard in input:
                logger.warning(f"Processing shard: {shard}")
                yield FastTextOperator._fasttext_filter.remote(
                    self.fasttext_url, self.input_column, self.filter_threshold, self.cache_dir, shard
                )

    @staticmethod
    @ray.remote
    def _fasttext_filter(
        fasttext_url: str, input_column: str, filter_threshold: float, cache_dir: str, data: Dataset
    ) -> Dataset:
        model = FastTextOperator._load_model(fasttext_url, cache_dir)
        if model is None:
            logger.warning(f"Failed to load model. Returning all data.")
            return data

        texts = data[input_column]
        texts = [" ".join(text.strip().split("\n")) for text in texts]
        probs = model.predict(texts, k=1)[1]
        scores = np.array([prob[0] for prob in probs])
        mask = scores > filter_threshold
        filtered_data = data.filter(lambda _, idx: mask[idx], with_indices=True)
        logger.info(f"Filtered {len(texts)} records to {len(filtered_data)} records")
        return filtered_data

    @staticmethod
    def _load_model(fasttext_url: str, cache_dir: str):
        cache_key = hashlib.md5(fasttext_url.encode()).hexdigest()

        success_file = os.path.join(cache_dir, f"{cache_key}.SUCCESS")
        model_dir = os.path.join(cache_dir, "model")
        model_file = os.path.join(model_dir, f"{cache_key}.bin")

        if os.path.exists(model_dir):
            logger.warning(f"Cache directory {cache_dir} already exists so will try to load model from cache.")
            model = FastTextOperator._wait_for_cache_or_download_model(model_file, success_file, fasttext_url)
            return model

        # Use a lock to prevent multiple processes from downloading the model at the same time.
        logger.warning(f"Acquiring lock on {cache_dir} to download model from {fasttext_url}")
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, f"{cache_key}.lock")
        fd = _acquire_lock(lock_path)

        if fd is not None:
            try:
                logger.warning(f"First process downloading model from {fasttext_url} to {model_dir}")
                download_file(fasttext_url, model_file)
                with open(success_file, "w") as f:
                    f.write("Success.")
                return fasttext.load_model(model_file)
            finally:
                _release_lock(fd, lock_path)

        # If we get here, we failed to acquire the lock, so we need to wait for the model to be downloaded to
        # cache by another process or download it ourselves.
        return FastTextOperator._wait_for_cache_or_download_model(model_file, success_file, fasttext_url)

    @staticmethod
    def _wait_for_cache_or_download_model(
        model_file: str, success_file: str, fasttext_url: str, timeout_threshold: int = 15
    ):
        # We reuse model from cache whenever possible. However, we don't want to a process deadlock waiting for cache,
        # so we download it ourselves if we time out waiting for the model.
        total_sleep = 0

        while not os.path.exists(success_file) and total_sleep <= timeout_threshold:
            time.sleep(1)
            logger.warning(f"Waiting for model to download. Total sleep: {total_sleep}")
            total_sleep += 1

        if os.path.exists(success_file):
            return fasttext.load_model(model_file)

        logger.warning(f"Failed waiting for model to be downloaded to cache. Will download to temp file.")

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                model_file = os.path.join(temp_dir, "model.bin")
                download_file(fasttext_url, model_file)
                return fasttext.load_model(model_file)
            except Exception as e:
                logger.error(f"Failed to download and load model: {str(e)}")
                return None


def _acquire_lock(lock_file):
    fd = os.open(lock_file, os.O_WRONLY | os.O_CREAT)

    try:
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Lock acquired on {lock_file}")
        return fd
    except IOError:
        # Another process has the lock
        print(f"Unable to acquire lock on {lock_file}. Another process has it.")
        os.close(fd)
        return None


def _release_lock(fd, lock_file):
    fcntl.lockf(fd, fcntl.LOCK_UN)
    os.close(fd)
    print(f"Lock released on {lock_file}")


def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as file:
        file.write(response.content)
    print(f"File '{filename}' has been downloaded successfully.")
