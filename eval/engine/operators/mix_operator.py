import random
from typing import List, Literal

import ray
from pydantic import Field

from datasets import Dataset, concatenate_datasets
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefsGenerator,
    Operator,
    OperatorSpecificConfig,
    ShardRef,
)


class MixOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for mix operators.

    Attributes:
        type (Literal["mix"]): The type of the operator, always set to "mix".
        seed (int): The seed for random shuffling (optional).
    """

    type: Literal["mix"] = "mix"
    seed: int = Field(default=42)


class MixOperator(Operator):
    """
    Operator that mixes incoming shards by concatenating and shuffling them.

    Attributes:
        seed (int): The seed for random shuffling.
    """

    def __init__(self, id: str, input_ids: List[str], config: MixOperatorConfig):
        """
        Initialize the MixOperator.

        Args:
            id (str): Unique identifier for the operator.
            input_ids (List[str]): List of input identifiers for the operator.
            config (MixOperatorConfig): Specific configuration for the mix operator.
        """
        super().__init__(id, input_ids, config)
        self.seed = config.seed

    def compute(self, inputs: DatasetRefs) -> ManyShardRefsGenerator:
        """
        Execute the mix operator on the input datasets.

        Args:
            inputs (DatasetRefs): Map of input datasets to mix

        Returns:
            ManyShardRefsGenerator: Generator of mixed and shuffled datasets
        """
        all_shards = []
        for input_shards in inputs.values():
            all_shards.extend(input_shards)

        yield self.mix_and_shuffle.remote(self, all_shards)

    @ray.remote
    def mix_and_shuffle(self, shards: List[ShardRef]) -> Dataset:
        """
        Mix and shuffle the input shards.

        Args:
            shards (List[ShardRef]): List of dataset shard references.

        Returns:
            Dataset: Mixed and shuffled dataset.
        """
        datasets = [ray.get(shard) for shard in shards]
        combined_dataset = concatenate_datasets(datasets)
        return combined_dataset.shuffle(seed=self.seed)
