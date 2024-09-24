from typing import List

import pandas as pd
import ray

from datasets import Dataset, concatenate_datasets
from engine.dataset import DatasetRef, DatasetRefs
from engine.operators.operator import Operator, OperatorSpecificConfig


class MapOperator(Operator):
    def execute(self, inputs: DatasetRefs) -> DatasetRef:
        outputs = []
        for input in inputs.values():
            for shard in input:
                processed_dataset = self.process_shard.remote(self, shard)
                outputs.append(processed_dataset)
        return outputs

    @ray.remote
    def process_shard(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError("Subclasses must implement process_shard method")

class ReduceOperator(Operator):
    def execute(self, inputs: DatasetRefs) -> DatasetRef:
        outputs = []
        for input in inputs.values():
            processed_dataset = self.process_shard.remote(self, [shard for shard in input])
            outputs.append(processed_dataset)
        return outputs

    @ray.remote
    def process_shard(self, datasets: List[Dataset]) -> Dataset:
        raise NotImplementedError("Subclasses must implement process_shard method")
