import gc
import json
import logging
import os
import random
import tempfile
from typing import List, Literal

import boto3
import ray
import zstandard as zstd
from botocore.exceptions import ClientError

from datasets import Dataset
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefsGenerator,
    Operator,
    OperatorSpecificConfig,
    register_operator,
)

logger = logging.getLogger(__name__)


class DCLMRefineWebSourceConfig(OperatorSpecificConfig):
    """
    Configuration class for DCLM RefinedWeb source operators.

    Attributes:
        type (Literal["dclm_refinedweb_source"]): The type of the operator.
        s3_bucket (str): The S3 bucket containing the data.
        s3_prefix (str): The S3 prefix for the data.
        num_shards (int): The number of shards to process.
        seed (int): Seed for random shard selection.
        chunk_size (int): Default chunk size for processing.
    """

    type: Literal["dclm_refinedweb_source"] = "dclm_refinedweb_source"
    s3_bucket: str
    s3_prefix: str
    num_shards: int
    seed: int = 42
    max_retries: int = 7
    base_delay: int = 3


class DCLMRefineWebSourceOperator(Operator):
    """
    Operator that loads and processes data from DCLM RefinedWeb source.

    Attributes:
        s3_bucket (str): The S3 bucket containing the data.
        s3_prefix (str): The S3 prefix for the data.
        num_shards (int): The number of shards to process.
        seed (int): Seed for random shard selection.
        max_retries (int): Maximum number of retries for S3 operations.
        base_delay (int): Base delay for S3 operations.
    """

    def __init__(self, id: str, input_ids: List[str], config: DCLMRefineWebSourceConfig):
        """
        Initialize the DCLMRefineWebSourceOperator.

        Args:
            id (str): Unique identifier for the operator.
            input_ids (List[str]): List of input identifiers for the operator.
            config (DCLMRefineWebSourceConfig): Specific configuration for the operator.
        """
        super().__init__(id, input_ids, config)
        self.s3_bucket = config.s3_bucket
        self.s3_prefix = config.s3_prefix
        self.num_shards = config.num_shards
        self.seed = config.seed
        self.max_retries = config.max_retries
        self.base_delay = config.base_delay

    def compute(self, _: DatasetRefs) -> ManyShardRefsGenerator:
        """
        Execute the DCLM RefinedWeb source operator to load and process the data.

        Args:
            _ (DatasetRefs): Unused input (for compatibility with the Operator interface).

        Returns:
            ManyShardRefs: List of references to the processed shards.
        """
        logger.info(f"Num shards: {self.num_shards}")
        shard_infos = self._get_shard_infos()
        logger.info(f"Processing {len(shard_infos)} shards")

        for shard_info in shard_infos:
            yield self._get_data_from_shard.remote(self.s3_bucket, shard_info, self.max_retries, self.base_delay)

    def _get_shard_infos(self) -> List[dict]:
        """Get the shard infos from the S3 bucket."""
        s3 = boto3.client("s3")
        random.seed(self.seed)
        shard_infos = []
        while len(shard_infos) < self.num_shards:
            global_shard = random.randint(1, 10)
            local_shard = random.randint(0, 9)

            prefix = f"{self.s3_prefix}global-shard_{global_shard:02d}_of_10/local-shard_{local_shard}_of_10/"

            local_shard_infos = self._get_local_shard_infos(s3, prefix)
            shard_infos.extend(local_shard_infos)

            if len(shard_infos) >= self.num_shards:
                return shard_infos[: self.num_shards]

        return shard_infos

    def _get_local_shard_infos(self, s3, prefix: str) -> List[dict]:
        """Get the infos about the compressed jsonl shards within a local shard from the S3 bucket."""
        local_shard_infos = []
        continuation_token = None

        while True:
            list_kwargs = {
                "Bucket": self.s3_bucket,
                "Prefix": prefix,
            }
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = s3.list_objects_v2(**list_kwargs)

            for obj in response.get("Contents", []):
                if obj["Key"].endswith("_processed.jsonl.zstd"):
                    shard_info = {"bucket": self.s3_bucket, "key": obj["Key"]}
                    if shard_info not in local_shard_infos:
                        local_shard_infos.append(shard_info)

            if not response.get("IsTruncated"):  # No more objects to list
                break

            continuation_token = response.get("NextContinuationToken")

        return local_shard_infos

    @staticmethod
    @ray.remote
    def _get_data_from_shard(bucket: str, shard_info: dict, max_retries: int, base_delay: int) -> Dataset:
        """
        Get data from a shard.
        """
        import time

        s3_client = boto3.client("s3")

        # Random sleep to avoid S3 throttling
        time.sleep(random.random() * 1)

        for attempt in range(max_retries):
            try:
                logger.warning(f"Attempting to process shard: {shard_info['key']}")
                response = s3_client.get_object(Bucket=bucket, Key=shard_info["key"])
                compressed_data = response["Body"].read()
                dctx = zstd.ZstdDecompressor()
                decompressed_data = dctx.decompress(compressed_data)

                chunk = []
                for line in decompressed_data.splitlines():
                    chunk.append(json.loads(line))

                dataset_chunk = Dataset.from_list(chunk)
                logger.warning(f"Yielded dataset for shard: {shard_info['key']}")
                return dataset_chunk
            except ClientError as e:
                if e.response["Error"]["Code"] == "SlowDown":
                    if attempt < max_retries - 1:
                        delay = (2**attempt) * base_delay
                        logging.warning(
                            f"Rate limited, retrying in {delay} seconds. Attempt {attempt + 1}/{max_retries}"
                        )
                        time.sleep(delay)


register_operator(DCLMRefineWebSourceConfig, DCLMRefineWebSourceOperator)
