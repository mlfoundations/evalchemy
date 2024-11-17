from typing import List, Literal

import ray

from datasets import Dataset
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefsGenerator,
    Operator,
    OperatorSpecificConfig,
)


class RenameColumnOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for Hugging Face dataset source operators.

    Attributes:
        type (Literal["rename_column"]): The type of the operator, always set to "rename_column".
        input_column_name (str): The name of the column to be renamed
        output_column_name (str): New column name
    """

    type: Literal["rename_column"] = "rename_column"
    input_column_name: str
    output_column_name: str


class RenameColumnOperator(Operator):
    """
    Operator that loads a dataset from Hugging Face's datasets library.

    Attributes:
        dataset (str): Name of the dataset to load.
        split (str): The split of the dataset to use.
        columns (Optional[List[str]]): Specific columns to load from the dataset.
        num_truncate (Optional[int]): Number of samples to truncate the dataset to.
    """

    def __init__(self, id: str, input_ids: List[str], config: RenameColumnOperatorConfig):
        """
        Initialize the RenameColumnOperator.

        Args:
            id (str): Unique identifier for the operator.
            input_ids (List[str]): List of input identifiers for the operator.
            config (RenameColumnOperatorConfig): Specific configuration for the HF source operator.
        """
        super().__init__(id, input_ids, config)
        self.input_column_name = config.input_column_name
        self.output_column_name = config.output_column_name

    def compute(self, inputs: DatasetRefs) -> ManyShardRefsGenerator:
        """
        Renames the column

        Args:
            inputs (DatasetRefs): input dataset

        Returns:
            ManyShardRefsGenerator: Generator of renamed column datasets.
        """
        input_shards = []

        for list_datasets in inputs.values():
            input_shards.extend(list_datasets)

        for input_dataset in input_shards:
            yield self.rename_column.remote(input_dataset, self.output_column_name, self.input_column_name)

    @ray.remote
    def rename_column(input_dataset: Dataset, output_column_name: str, input_column_name: str) -> Dataset:
        """
        Renames the column

        Args:
            inputs (DatasetRefs): input dataset

        Returns:
            Dataset: Renamed column dataset
        """
        column_names = input_dataset.column_names

        if output_column_name in column_names:
            input_dataset = input_dataset.remove_columns(output_column_name)

        name_mapping = {input_column_name: output_column_name}
        renamed_dataset = input_dataset.rename_columns(name_mapping)
        return renamed_dataset
