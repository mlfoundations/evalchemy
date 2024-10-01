import json
import logging
import random
import time
from typing import List, Literal

import boto3
import ray
import zstandard as zstd
from botocore.exceptions import ClientError

from datasets import Dataset
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefs,
    Operator,
    OperatorSpecificConfig,
    register_operator,
)


class DCLMRefineWebSourceConfig(OperatorSpecificConfig):
    """
    Configuration class for DCLM RefinedWeb source operators.

    Attributes:
        type (Literal["dclm_refinedweb_source"]): The type of the operator.
        s3_bucket (str): The S3 bucket containing the data.
        s3_prefix (str): The S3 prefix for the data.
        num_shards (int): The number of shards to process.
        seed (int): Seed for random shard selection.
    """

    type: Literal["dclm_refinedweb_source"] = "dclm_refinedweb_source"
    s3_bucket: str
    s3_prefix: str
    num_shards: int
    seed: int = 42


class DCLMRefineWebSourceOperator(Operator):
    """
    Operator that loads and processes data from DCLM RefinedWeb source.

    Attributes:
        s3_bucket (str): The S3 bucket containing the data.
        s3_prefix (str): The S3 prefix for the data.
        num_shards (int): The number of shards to process.
        seed (int): Seed for random shard selection.
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

    def execute(self, _: DatasetRefs) -> ManyShardRefs:
        """
        Execute the DCLM RefinedWeb source operator to load and process the data.

        Args:
            _ (DatasetRefs): Unused input (for compatibility with the Operator interface).

        Returns:
            ManyShardRefs: List of references to the processed shards.
        """
        shard_infos = self._get_shard_infos()
        logging.info(f"Shard infos: {shard_infos}")
        logging.info(f"Processing {len(shard_infos)} shards")
        return [self._process_shard.remote(self.s3_bucket, shard_info) for shard_info in shard_infos]

    def _get_shard_infos(self) -> List[dict]:
        s3 = boto3.client("s3")
        random.seed(self.seed)
        shard_infos = []

        while len(shard_infos) < self.num_shards:
            global_shard = random.randint(1, 10)
            local_shard = random.randint(0, 9)

            prefix = f"{self.s3_prefix}global-shard_{global_shard:02d}_of_10/local-shard_{local_shard}_of_10/"
            response = s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix)

            for obj in response.get("Contents", []):
                if obj["Key"].endswith("_processed.jsonl.zstd"):
                    shard_info = {"bucket": self.s3_bucket, "key": obj["Key"]}
                    if shard_info not in shard_infos:
                        shard_infos.append(shard_info)
                        break

            if len(shard_infos) >= self.num_shards:
                break

        return shard_infos

    @staticmethod
    @ray.remote
    def _process_shard(bucket: str, shard_info: dict) -> Dataset:
        s3_client = boto3.client("s3")
        max_retries = 7
        base_delay = 3

        # Random sleep to avoid S3 throttling
        time.sleep(random.random() * 1)

        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting to process shard: {shard_info['key']}")
                response = s3_client.get_object(Bucket=bucket, Key=shard_info["key"])
                compressed_data = response["Body"].read()

                dctx = zstd.ZstdDecompressor()
                decompressed_data = dctx.decompress(compressed_data)

                # Parse JSON lines and create a Dataset
                records = [json.loads(line) for line in decompressed_data.splitlines()]
                dataset = Dataset.from_list(records)

                logging.info(f"Successfully processed shard: {shard_info['key']}")
                return dataset
            except ClientError as e:
                if e.response["Error"]["Code"] == "SlowDown":
                    if attempt < max_retries - 1:
                        delay = (2**attempt) * base_delay
                        logging.warning(
                            f"Rate limited, retrying in {delay} seconds. Attempt {attempt + 1}/{max_retries}"
                        )
                        time.sleep(delay)
                    else:
                        logging.error(f"Max retries reached for shard: {shard_info['key']}")
                        raise
                else:
                    logging.error(f"Unexpected error for shard {shard_info['key']}: {str(e)}")
                    raise


register_operator(DCLMRefineWebSourceConfig, DCLMRefineWebSourceOperator)
