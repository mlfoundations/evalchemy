import importlib
import logging
from typing import Any, Callable, Dict, List, Literal

import pandas as pd
import ray
from pydantic import Field

from dataclasses import dataclass
from datasets import Dataset, concatenate_datasets
from engine.operators.configs import OperatorSpecificConfig
from engine.dataset import DatasetRef, DatasetRefs
from engine.operators.map import MapOperator, ReduceOperator
from engine.operators.registry import register_operator

@dataclass
class MergeOperatorConfig(OperatorSpecificConfig):
    type: Literal["merge"] = "merge"

@dataclass
class ShardOperatorConfig(OperatorSpecificConfig):
    type: Literal["shard"] = "shard"
    num_shards: int = 20

class MergeShardsOperator(ReduceOperator):
    def __init__(self, id: str, input_ids: List[str], config: MergeOperatorConfig):
        super().__init__(id, input_ids, config)
        self.config = config

    @ray.remote
    def process_shard(self, shards: List[Dataset]) -> Dataset:
        return concatenate_datasets(shards)
        

class ShardDatasetOperator(MapOperator):
    def __init__(self, id: str, input_ids: List[str], config: ShardOperatorConfig):
        super().__init__(id, input_ids, config)
        self.function = self._load_function(config.function)
        self.num_shards = config.num_shards

    @ray.remote
    def process_shard(self, dataset: Dataset) -> List[Dataset]:
        total_size = len(dataset)
        split_size = total_size // self.num_shards
        remainder = total_size % self.num_shards

        # Create the splits
        splits = []
        start = 0
        for i in range(self.num_shards):
            end = start + split_size + (1 if i < remainder else 0)
            splits.append(dataset.select(range(start, end)))
            start = end
        return splits

register_operator(MergeOperatorConfig, MergeShardsOperator)
register_operator(ShardOperatorConfig, ShardDatasetOperator)

