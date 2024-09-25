from abc import ABC, abstractmethod

from typing import Dict, List, TypeAlias, Callable
import importlib
import logging
import ray
from lm_eval.utils import eval_logger
from datasets import Dataset, concatenate_datasets


from engine.operators.registry import register_operator
from engine.operators.configs import OperatorSpecificConfig, OperatorConfig, FunctionOperatorConfig, HFSourceOperatorConfig

ShardRef: TypeAlias = ray.ObjectRef
ManyShardRefs: TypeAlias = List[ShardRef]
DatasetRefs: TypeAlias = Dict[str, ManyShardRefs]

class Operator(ABC):
    def __init__(self, id: str, input_ids: List[str], config: OperatorSpecificConfig):
        self._id = id
        self._input_ids = input_ids
        self._config = config

    @property
    def id(self) -> str:
        return self._id

    @property
    def input_ids(self) -> List[str]:
        return self._input_ids

    @property
    def config(self) -> OperatorSpecificConfig:
        return self._config

    @abstractmethod
    def execute(self, inputs: DatasetRefs) -> ManyShardRefs:
        pass

def create_operator(config: OperatorConfig) -> Operator:
    from engine.operators.registry import get_operator_class

    operator_class = get_operator_class(type(config.config))
    if operator_class is None:
        raise ValueError(f"Unknown operator type: {type(config.config)}")
    return operator_class(config.id, config.input_ids, config.config)


class FunctionOperator(Operator):
    def __init__(self, id: str, input_ids: List[str], config: FunctionOperatorConfig):
        super().__init__(id, input_ids, config)
        self.function = self._load_function(config.function)
        self.function_config = config.function_config
        self.num_shards = config.num_shards
        self.sharded = config.sharded

    def _load_function(self, function_path: str) -> Callable[[Dataset], Dataset]:
        module_name, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)

    def execute(self, inputs: DatasetRefs) -> ManyShardRefs:
        outputs = []
        input_shards = []
        for input in inputs.values():
            input_shards.extend(input)
        # Sharding stage
        if self.sharded and len(input_shards) == 1:
            shards_to_process = self.shard_dataset.remote(self, input_shards[0])
        elif not self.sharded and len(input_shards) > 1:
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
        split_size = total_size // self.num_shards
        remainder = total_size % self.num_shards

        # Create the splits
        splits = []
        start = 0
        for i in range(self.num_shards):
            end = start + split_size + (1 if i < remainder else 0)
            split = dataset.select(range(start, end))
            start = end
            yield split
        # TODO: does Ray wrap this as a Ref of list or list of Refs?

    @ray.remote
    def merge_shards(self, shards: ManyShardRefs) -> Dataset:
        return concatenate_datasets([ray.get(shard) for shard in shards])

    @ray.remote
    def process_shard(self, dataset: Dataset) -> Dataset:
        logging.info(f"Processing shard with function: {self.function.__name__}")
        return self.function(dataset, **self.function_config)


class HFSourceOperator(Operator):
    def __init__(self, id: str, input_ids: List[str], config: HFSourceOperatorConfig):
        super().__init__(id, input_ids, config)
        self.dataset = config.dataset
        self.split = config.split
        self.columns = config.columns
        self.num_truncate = config.num_truncate

    def execute(self, _: DatasetRefs) -> ManyShardRefs:
        dataset = self.load_dataset()
        return [ray.put(dataset)]

    def load_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset, split=self.split)
        if self.columns:
            dataset = dataset.select_columns(self.columns)
        if self.num_truncate is not None:
            dataset = dataset.select(range(min(len(dataset), self.num_truncate)))
        eval_logger.info(f"\nDataset loaded from {self.dataset}:")
        eval_logger.info(dataset)
        return dataset


register_operator(HFSourceOperatorConfig, HFSourceOperator)
register_operator(FunctionOperatorConfig, FunctionOperator)

