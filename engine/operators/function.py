import importlib
import logging
from typing import Any, Callable, Dict, List, Literal, Generator

import pandas as pd
import ray
from pydantic import Field

from datasets import Dataset, concatenate_datasets
from engine.operators.configs import OperatorSpecificConfig, FunctionOperatorConfig
from engine.dataset import DatasetRef, DatasetRefs
from engine.operators.map import MapOperator
from engine.operators.registry import register_operator
from engine.operators.operator import Operator




class FunctionOperator(Operator):
    def __init__(self, id: str, input_ids: List[str], config: FunctionOperatorConfig):
        super().__init__(id, input_ids, config)
        self.function = self._load_function(config.function)
        self.function_config = config.function_config

    def _load_function(self, function_path: str) -> Callable[[Dataset], Dataset]:
        module_name, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    
    def execute(self, inputs: DatasetRefs) -> DatasetRef:
        outputs = []
        input_shards = []
        for input in inputs.values():
            input_shards.extend(input)
        # Sharding stage
        if self.config.sharded and len(input_shards) == 1:
            shards_to_process = self.shard_dataset.remote(self, input_shards[0])
        elif not self.config.sharded and len(input_shards) > 1:
            shards_to_process = [self.merge_shards.remote(self, input_shards)]
        else:
            shards_to_process = input_shards

        # Processing stage: if dataset is sharded, then shards_to_process contains multiple shards
        # otherwise, it contains a single dataset.
        try:
            for shard in shards_to_process:
                processed_dataset = self.process_shard.remote(self, shard)
                outputs.append(processed_dataset)
        except:
            breakpoint()
        return outputs

    @ray.remote
    def shard_dataset(self, dataset: Dataset) -> ray.ObjectRefGenerator:
        # TODO: implement
        total_size = len(dataset)
        breakpoint()
        split_size = total_size // self.function_config.num_shards
        remainder = total_size % self.function_config.num_shards

        # Create the splits
        splits = []
        start = 0
        for i in range(self.function_config.num_shards):
            end = start + split_size + (1 if i < remainder else 0)
            split = dataset.select(range(start, end))
            start = end
            yield split
        # TODO: does Ray wrap this as a Ref of list or list of Refs?

    @ray.remote
    def merge_shards(self, shards: List[Dataset]) -> Dataset:
        return concatenate_datasets(shards)

    @ray.remote
    def process_shard(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError("Subclasses must implement process_shard method")


register_operator(FunctionOperatorConfig, FunctionOperator)
