import importlib
import inspect
import itertools
import logging
from itertools import chain
from typing import Any, Callable, Dict, Generator, List, Literal

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


class FunctionOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for function operators.

    Attributes:
        type (Literal["function"]): The type of the operator, always set to "function".
        function (str): The name or identifier of the function.
        function_config (Dict[str, Any]): Additional configuration for the function.
        sharded (bool): Indicates whether the function can operate across only a shard
        num_shards (int): The number of shards if the function is sharded.
        input_dataset_map (Dict[str, str]): Mapping of function argument names to input datasets from previous operators
    """

    type: Literal["function"] = "function"
    function: str
    function_config: Dict[str, Any] = Field(default_factory=dict)
    sharded: bool = False
    num_shards: int = 15
    input_dataset_map: Dict[str, str] = Field(default_factory=dict)


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
        self.input_dataset_map = config.input_dataset_map

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

    def compute(self, inputs: DatasetRefs) -> ManyShardRefsGenerator:
        """
        Execute the function operator on the input datasets.

        Args:
            inputs (DatasetRefs): Map of input datasets to apply function on

        Returns:
            ManyShardRefsGenerator: Generator of shards outputted by the function
        """
        sig = inspect.signature(self.function)
        parameters = list(sig.parameters.values())

        # Count the number of Dataset parameters in the function signature
        expected_datasets = [param for param in parameters if param.annotation == Dataset]

        if len(expected_datasets) == 1:
            # Single dataset case
            if len(inputs) > 1:
                logging.info(
                    f"Operator {self.id}: Expects 1 dataset, but more than one input_ids were provided. "
                    f"Will run function over all the shards from input datasets."
                )
            elif len(inputs) == 0:
                raise ValueError(f"Operator {self.id}: Expects 1 dataset, but no input_ids were provided.")

            arg_name = expected_datasets[0].name
            input_dataset = itertools.chain(*[iter(input_dataset) for input_dataset in inputs.values()])

            is_dataset_sharded = False
            first_element = next(input_dataset, None)
            if first_element is None:
                raise ValueError(f"Operator {self.id}: No shards found in input dataset.")

            second_element = next(input_dataset, None)
            # If there's a second shard, then the dataset is sharded
            is_dataset_sharded = second_element is not None

            input_shards = (
                chain([first_element, second_element], input_dataset) if is_dataset_sharded else [first_element]
            )

            if self.sharded and not is_dataset_sharded:
                shards_to_process = self.shard_dataset.options(name=f"sharding::{self.function.__name__}").remote(
                    first_element, self.num_shards
                )
            elif not self.sharded and is_dataset_sharded:
                shards_to_process = [
                    self.merge_shards.options(name=f"merging::{self.function.__name__}").remote(list(input_shards))
                ]
            else:
                shards_to_process = input_shards

            for shard in shards_to_process:
                processed_dataset = self.process_with_dataset.options(name=self.function.__name__).remote(
                    {arg_name: shard}, self.function, self.function_config
                )
                yield processed_dataset

        elif len(expected_datasets) > 1:
            # Multiple datasets case
            if len(inputs) != len(expected_datasets):
                raise ValueError(
                    f"Operator {self.id}: Function expects {len(expected_datasets)} datasets, but {len(inputs)} were provided."
                )

            if self.sharded:
                raise ValueError(f"Operator {self.id}: Function with multiple sources of inputs cannot be sharded.")

            if len(self.input_dataset_map) == 0:
                raise ValueError(
                    f"Operator {self.id}: More than one dataset needed in function, but 'input_dataset_map' is not set!"
                )

            if len(self.input_dataset_map) != len(expected_datasets):
                raise ValueError(
                    f"Operator {self.id}: Length of input_dataset_map does not match the number of datasets needed."
                )
            mapped_inputs = {
                arg: self.merge_shards.remote(list(inputs[key]))
                for arg, key in self.input_dataset_map.items()
                if arg in sig.parameters
            }

            result = self.process_with_dataset.options(name=self.function.__name__).remote(
                mapped_inputs, self.function, self.function_config
            )
            yield result

        elif len(expected_datasets) == 0:
            # No datasets case (source function)
            result = self.process_without_dataset.options(name=self.function.__name__).remote(
                self.function, self.function_config
            )
            yield result
        else:
            raise ValueError(f"Operator {self.id}: Unexpected number of Dataset parameters: {len(expected_datasets)}")

    @staticmethod
    @ray.remote
    def shard_dataset(dataset: Dataset, num_shards: int) -> ray.ObjectRefGenerator:
        """
        Shard the input dataset into multiple parts to utilize parallelism.

        Args:
            dataset (Dataset): The input dataset to be sharded.
            num_shards (int): The number of shards to create.

        Returns:
            ray.ObjectRefGenerator: Generator of dataset shards.
        """
        total_size = len(dataset)
        split_size = max(total_size // num_shards, 1)

        start = 0
        while start < total_size:
            end = start + split_size
            split = dataset.select(range(start, min(end, total_size)))
            start = end
            yield split

    @staticmethod
    @ray.remote
    def merge_shards(shards: List[ShardRef]) -> Dataset:
        """
        Merge multiple dataset shards into a single dataset if function requires all data at once.

        Args:
            shards (List[ShardRef]): List of dataset shard references.

        Returns:
            Dataset: Merged dataset.
        """
        return concatenate_datasets([ray.get(shard) for shard in shards])

    @staticmethod
    @ray.remote
    def process_without_dataset(function: Callable, function_config: Dict[str, Any]) -> Any:
        """
        Process using the configured function without passing a dataset.
        """
        logging.info(f"Processing with function: {function.__name__}")
        return function(**function_config)

    @staticmethod
    @ray.remote
    def process_with_dataset(
        mapped_inputs: Dict[str, ShardRef], function: Callable, function_config: Dict[str, Any]
    ) -> Dataset:
        """
        Process datasets using the configured function.

        Args:
            mapped_inputs (Dict[str, ShardRef]): A dictionary mapping parameter names to shard references (merged in previous step)
            function (Callable): The function to apply to the datasets
            function_config (Dict[str, Any]): Additional configuration for the function

        Returns:
            Dataset: The result of applying the function to the input datasets.
        """
        logging.info(f"Processing multiple datasets with function: {function.__name__}")

        processed_mapped_inputs = {k: ray.get(v) for k, v in mapped_inputs.items()}
        all_inputs = {**function_config, **processed_mapped_inputs}

        return function(**all_inputs)
