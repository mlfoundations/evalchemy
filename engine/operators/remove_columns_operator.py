from typing import List, Literal

import ray

from datasets import Dataset
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefsGenerator,
    Operator,
    OperatorSpecificConfig,
)


class RemoveColumnsOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for RemoveColumn operators.

    Attributes:
        type (Literal["remove_columns"]): The type of the operator, always set to "remove_columns".
        columns_to_keep (List[str]): List of columns to keep
    """

    type: Literal["remove_columns"] = "remove_columns"
    columns_to_keep: List[str]


class RemoveColumnsOperator(Operator):
    """
    Operator that loads a dataset from Hugging Face's datasets library.

    Attributes:
        dataset (str): Name of the dataset to load.
        split (str): The split of the dataset to use.
        columns (Optional[List[str]]): Specific columns to load from the dataset.
        num_truncate (Optional[int]): Number of samples to truncate the dataset to.
    """

    def __init__(self, id: str, input_ids: List[str], config: RemoveColumnsOperatorConfig):
        """
        Initialize the RemoveColumnsOperator.

        Args:
            id (str): Unique identifier for the operator.
            input_ids (List[str]): List of input identifiers for the operator.
            config (RemoveColumnsOperatorConfig): Specific configuration for the HF source operator.
        """
        super().__init__(id, input_ids, config)
        self.columns_to_keep = config.columns_to_keep

    def compute(self, inputs: DatasetRefs) -> ManyShardRefsGenerator:
        """
        Renames the column

        Args:
            inputs (DatasetRefs): input dataset

        Returns:
            ManyShardRefsGenerator: Generator of removed column datasets.
        """
        input_shards = []

        for list_datasets in inputs.values():
            input_shards.extend(list_datasets)

        for input_dataset in input_shards:
            yield self.remove_columns.remote(input_dataset, self.columns_to_keep)

    @staticmethod
    @ray.remote
    def remove_columns(input_dataset: Dataset, columns_to_keep: List[str]) -> Dataset:
        """
        Renames the column

        Args:
            inputs (DatasetRefs): input dataset

        Returns:
            Dataset: Renamed column dataset
        """
        all_columns = input_dataset.column_names
        columns_to_remove = [col for col in all_columns if col not in columns_to_keep]
        cleaned_dataset = input_dataset.remove_columns(columns_to_remove)
        return cleaned_dataset
