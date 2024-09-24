import logging
from typing import List, Literal, Optional

import ray

from datasets import Dataset, load_dataset
from engine.operators.configs import OperatorSpecificConfig
from engine.dataset import DatasetRef, DatasetRefs
from engine.operators.operator import Operator
from engine.operators.registry import register_operator
from lm_eval.utils import eval_logger

class HFSourceOperatorConfig(OperatorSpecificConfig):
    type: Literal["hf_source"] = "hf_source"
    dataset: str
    split: str
    columns: Optional[List[str]] = None
    num_truncate: Optional[int] = None


class HFSourceOperator(Operator):
    def __init__(self, id: str, input_ids: List[str], config: HFSourceOperatorConfig):
        super().__init__(id, input_ids, config)
        self.dataset = config.dataset
        self.split = config.split
        self.columns = config.columns
        self.num_truncate = config.num_truncate

    def execute(self, _: DatasetRefs) -> DatasetRef:
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
