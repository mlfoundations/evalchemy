import importlib
import inspect
import logging
from typing import Any, Callable, Dict, List, Literal

import ray
from pydantic import Field

from datasets import Dataset, concatenate_datasets
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefs,
    Operator,
    OperatorSpecificConfig,
)


class FunctionOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for function operators.

    Attributes:
        type (Literal["function"]): The type of the operator, always set to "function".
        function (str): The name or identifier of the function.
        function_config (Dict[str, Any]): Additional configuration for the function.
        sharded (bool): Indicates whether the function can operate across only a shard
        num_shards (int): The number of shards if the function is sharded.
    """

    type: Literal["function"] = "function"
    function: str
    function_config: Dict[str, Any] = Field(default_factory=dict)
    sharded: bool = False
    num_shards: int = 3


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

    def execute(self, inputs: DatasetRefs) -> ManyShardRefs:
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

        sig = inspect.signature(self.function)
        parameters = list(sig.parameters.values())

        # If the function takes a Dataset as its first argument,
        # process the shards, otherwise this means the function
        # doesn't need a Dataset as input and is a "source"
        # so we call it without an input dataset.
        if parameters and parameters[0].annotation == Dataset:
            # If it is, process shards as before
            if self.sharded and len(input_shards) == 1:
                shards_to_process = self.shard_dataset.remote(self, input_shards[0])
            elif not self.sharded and len(input_shards) > 1:
                shards_to_process = [self.merge_shards.remote(self, input_shards)]
            else:
                shards_to_process = input_shards

            for shard in shards_to_process:
                processed_dataset = self.process_shard_with_dataset.remote(self, shard)
                outputs.append(processed_dataset)
        else:
            result = self.process_without_dataset.remote(self)
            outputs.append(result)

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
        total_size = len(dataset)
        split_size = total_size // self.num_shards
        remainder = total_size % self.num_shards

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
    def process_shard_with_dataset(self, dataset: Dataset) -> Dataset:
        """
        Process a single dataset or single shard using the configured function.
        """
        logging.info(f"Processing shard with function: {self.function.__name__}")
        return self.function(dataset, **self.function_config)

    @ray.remote
    def process_without_dataset(self) -> Any:
        """
        Process using the configured function without passing a dataset.
        """
        logging.info(f"Processing with function: {self.function.__name__}")
        return self.function(**self.function_config)
