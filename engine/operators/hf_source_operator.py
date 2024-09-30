import logging
from typing import List, Literal, Optional

import ray

from datasets import Dataset, load_dataset
from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefs,
    Operator,
    OperatorSpecificConfig,
    register_operator,
)


class HFSourceOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for Hugging Face dataset source operators.

    Attributes:
        type (Literal["hf_source"]): The type of the operator, always set to "hf_source".
        dataset (str): The name of the Hugging Face dataset.
        split (str): The split of the dataset to use.
        columns (Optional[List[str]]): Specific columns to load from the dataset.
        num_truncate (Optional[int]): Number of samples to truncate the dataset to.
    """

    type: Literal["hf_source"] = "hf_source"
    dataset: str
    split: str
    columns: Optional[List[str]] = None
    num_truncate: Optional[int] = None


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

    def execute(self, _: DatasetRefs) -> ManyShardRefs:
        """
        Execute the HF source operator to load the dataset.

        Args:
            _ (DatasetRefs): Unused input (for compatibility with the Operator interface).

        Returns:
            ManyShardRefs: List containing a single reference to the loaded dataset.
        """
        dataset = self.load_dataset()
        return [ray.put(dataset)]

    def load_dataset(self) -> Dataset:
        """
        Load the dataset from Hugging Face's datasets library.

        Returns:
            Dataset: The loaded and potentially processed dataset.
        """
        # The keep_in_memory flag being set to True is crucial to allow us
        # to store the Dataset, along with its actual content, in Ray's object store.
        # Otherwise, Dataset only contains pointers to Arrow Tables written to disk.
        dataset = load_dataset(self.dataset, split=self.split, keep_in_memory=True)
        if self.columns:
            dataset = dataset.select_columns(self.columns)
        if self.num_truncate is not None:
            dataset = dataset.select(range(min(len(dataset), self.num_truncate)))
        logging.info(f"\nDataset loaded from {self.dataset}:")
        logging.info(dataset)
        return dataset
