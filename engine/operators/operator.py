from abc import ABC, abstractmethod
import fsspec
import os
from typing import Dict, List, TypeAlias, Callable, Type
import importlib
import logging
from functools import partial

import ray
from lm_eval.utils import eval_logger
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

from engine.operators.configs import (
    OperatorSpecificConfig,
    OperatorConfig,
    FunctionOperatorConfig,
    HFSourceOperatorConfig,
    CONFIG_TYPE_MAP,
)

ShardRef: TypeAlias = ray.ObjectRef
ManyShardRefs: TypeAlias = List[ShardRef]
DatasetRefs: TypeAlias = Dict[str, ManyShardRefs]


class Operator(ABC):
    """
    Abstract base class for all operators in the data processing pipeline.

    Attributes:
        id (str): Unique identifier for the operator.
        input_ids (List[str]): List of input identifiers for the operator.
        config (OperatorSpecificConfig): Specific configuration for the operator.
    """

    def __init__(self, id: str, input_ids: List[str], config: OperatorSpecificConfig):
        self._id = id
        self._input_ids = input_ids
        self._config = config
        self.cache_dir = None

    @property
    def id(self) -> str:
        """Get the operator's unique identifier."""
        return self._id

    @property
    def input_ids(self) -> List[str]:
        """Get the list of input identifiers for the operator."""
        return self._input_ids

    @property
    def config(self) -> OperatorSpecificConfig:
        """Get the specific configuration for the operator."""
        return self._config

    def execute(self, inputs: DatasetRefs) -> ManyShardRefs:
        self.outputs = self.compute(inputs)
        return self.outputs

    @abstractmethod
    def compute(self, inputs: DatasetRefs) -> ManyShardRefs:
        """
        compute the operator on the given inputs.

        Args:
            inputs (DatasetRefs): Dictionary of inputs mapping identifiers to a list of shard references (known as a dataset)

        Returns:
            ManyShardRefs: List of processedoutput shard references for each input shard
        """
        pass

    def cleanup(self, fs: fsspec.AbstractFileSystem, overwrite_cache: bool = False, cache_dir: str = None):
        if not fs.exists(cache_dir) and cache_dir is None:
            raise ValueError(f"Cache Directory of {self._id} not set")

        if fs.exists(cache_dir) and not overwrite_cache:
            return

        if not fs.exists(cache_dir):
            fs.mkdir(cache_dir, create_parents=True)

        items_to_save = [ray.get(output) for output in self.outputs]
        custom_open = partial(fs.open)
        for idx, item in enumerate(items_to_save):
            item.save_to_disk(f"{cache_dir}/{idx}.hf", storage_options={"open": custom_open})


def create_operator(config: OperatorConfig) -> Operator:
    """
    Create an operator instance based on the given configuration.

    Args:
        config (OperatorConfig): Configuration for the operator.

    Returns:
        Operator: An instance of the appropriate Operator subclass.

    Raises:
        ValueError: If the operator type is unknown.
    """
    operator_class = get_operator_class(type(config.config))
    if operator_class is None:
        raise ValueError(f"Unknown operator type: {type(config.config)}")
    return operator_class(config.id, config.input_ids, config.config)


class LoadFromCacheOperator(Operator):
    def __init__(self, id: str, input_ids: List[str], cache_dir: str):
        super().__init__(id, input_ids, None)
        self.cache_dir = cache_dir

    def compute(self, _: DatasetRefs) -> ManyShardRefs:
        # Get all .hf files in the cache directory
        dataset_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".hf")]

        # Load datasets and create a list with ray.put
        datasets = []
        for file in dataset_files:
            file_path = os.path.join(self.cache_dir, file)
            dataset = load_from_disk(file_path)
            datasets.append(ray.put(dataset))

        return datasets


class FunctionOperator(Operator):
    """
    Operator that applies a function to the input dataset or shard.

    Attributes:
        function (Callable[[Dataset], Dataset]): The function to apply to the dataset or shard (shard of a dataset is a dataset).
        function_config (Dict[str, Any]): Additional configuration for the function.
        num_shards (int): Number of shards to split the dataset into if the function can operate across individual shards
        sharded (bool): If the function can be applied to individual shards of a dataset rather than the whole, set this to true to utilize parallelism
    """

    def __init__(self, id: str, input_ids: List[str], config: FunctionOperatorConfig):
        """
        Initialize the FunctionOperator.

        Args:
            id (str): Unique identifier for the operator.
            input_ids (List[str]): List of input identifiers for the operator.
            config (FunctionOperatorConfig): Specific configuration for the function operator.
        """
        super().__init__(id, input_ids, config)
        self.function = self._load_function(config.function)
        self.function_config = config.function_config
        self.num_shards = config.num_shards
        self.sharded = config.sharded

    def _load_function(self, function_path: str) -> Callable[[Dataset], Dataset]:
        """
        Load the function from the given path.

        Args:
            function_path (str): Path to the function.

        Returns:
            Callable[[Dataset], Dataset]: The loaded function.
        """
        module_name, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)

    def compute(self, inputs: DatasetRefs) -> ManyShardRefs:
        """
        Execute the function operator on the input datasets.

        Args:
            inputs (DatasetRefs): Map of input datasets to apply function on

        Returns:
            ManyShardRefs: List of shards outputted by the function for each input
        """
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

        for shard in shards_to_process:
            processed_dataset = self.process_shard.remote(self, shard)
            outputs.append(processed_dataset)

        self.outputs = outputs
        return outputs

    @ray.remote
    def shard_dataset(self, dataset: Dataset) -> ray.ObjectRefGenerator:
        """
        Shard the input dataset into multiple parts to utilize parallelism.

        Args:
            dataset (Dataset): The input dataset to be sharded.

        Returns:
            ray.ObjectRefGenerator: Generator of dataset shards.
        """
        # TODO: implement
        total_size = len(dataset)
        split_size = total_size // self.num_shards
        remainder = total_size % self.num_shards

        # Create the splits
        start = 0
        for i in range(self.num_shards):
            end = start + split_size + (1 if i < remainder else 0)
            split = dataset.select(range(start, end))
            start = end
            yield split

    @ray.remote
    def merge_shards(self, shards: ManyShardRefs) -> Dataset:
        """
        Merge multiple dataset shards into a single dataset if function requires all data at once.

        Args:
            shards (ManyShardRefs): List of dataset shard references.

        Returns:
            Dataset: Merged dataset.
        """
        return concatenate_datasets([ray.get(shard) for shard in shards])

    @ray.remote
    def process_shard(self, dataset: Dataset) -> Dataset:
        """
        Process a single dataset or single shard (a shard is a dataset) using the configured function.

        Args:
            dataset (Dataset): The input dataset or single shard (a shard is a dataset)

        Returns:
            Dataset: Processed dataset or shard.
        """
        logging.info(f"Processing shard with function: {self.function.__name__}")
        return self.function(dataset, **self.function_config)


class HFSourceOperator(Operator):
    """
    Operator that loads a dataset from Hugging Face's datasets library.

    Attributes:
        dataset (str): Name of the dataset to load.
        split (str): The split of the dataset to use.
        columns (Optional[List[str]]): Specific columns to load from the dataset.
        num_truncate (Optional[int]): Number of samples to truncate the dataset to.
    """

    def __init__(self, id: str, input_ids: List[str], config: HFSourceOperatorConfig):
        """
        Initialize the HFSourceOperator.

        Args:
            id (str): Unique identifier for the operator.
            input_ids (List[str]): List of input identifiers for the operator.
            config (HFSourceOperatorConfig): Specific configuration for the HF source operator.
        """

        super().__init__(id, input_ids, config)
        self.dataset = config.dataset
        self.split = config.split
        self.columns = config.columns
        self.num_truncate = config.num_truncate

    def compute(self, _: DatasetRefs) -> ManyShardRefs:
        """
        Execute the HF source operator to load the dataset.

        Args:
            _ (DatasetRefs): Unused input (for compatibility with the Operator interface).

        Returns:
            ManyShardRefs: List containing a single reference to the loaded dataset.
        """
        dataset = self.load_dataset()
        self.outputs = [ray.put(dataset)]
        return self.outputs

    def load_dataset(self) -> Dataset:
        """
        Load the dataset from Hugging Face's datasets library.

        Returns:
            Dataset: The loaded and potentially processed dataset.
        """
        dataset = load_dataset(self.dataset, split=self.split)
        if self.columns:
            dataset = dataset.select_columns(self.columns)
        if self.num_truncate is not None:
            dataset = dataset.select(range(min(len(dataset), self.num_truncate)))
        eval_logger.info(f"\nDataset loaded from {self.dataset}:")
        eval_logger.info(dataset)
        return dataset


def register_operator(config_class: Type[OperatorSpecificConfig], operator_class: Type[Operator]):
    """
    Register an operator class with its corresponding configuration class.

    Args:
        config_class (Type[OperatorSpecificConfig]): The configuration class for the operator.
        operator_class (Type[Operator]): The operator class to be registered.
    """
    OPERATOR_MAP[config_class] = operator_class
    CONFIG_TYPE_MAP[config_class.model_fields["type"].default] = config_class


def get_operator_class(config_class: Type[OperatorSpecificConfig]) -> Type[Operator]:
    """
    Get the operator class corresponding to a given configuration class.

    Args:
        config_class (Type[OperatorSpecificConfig]): The configuration class to look up.

    Returns:
        Type[Operator]: The corresponding operator class, or None if not found.
    """
    return OPERATOR_MAP.get(config_class)


OPERATOR_MAP: Dict[Type[OperatorSpecificConfig], Type[Operator]] = {}

register_operator(HFSourceOperatorConfig, HFSourceOperator)
register_operator(FunctionOperatorConfig, FunctionOperator)
