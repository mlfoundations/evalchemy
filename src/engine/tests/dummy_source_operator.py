from typing import List, Literal

import ray
from pydantic import Field

from datasets import Dataset
from engine.operators.operator import (
    DatasetRefs,
    Operator,
    OperatorSpecificConfig,
    ShardRef,
    register_operator,
)


class DummySourceOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for the dummy source operator.

    Attributes:
        type (Literal["dummy_source"]): The type of the operator, always set to "dummy_source".
        num_rows (int): The number of rows in each shard. Defaults to 100.
    """

    type: Literal["dummy_source"] = "dummy_source"
    num_rows: int = Field(default=5, ge=1)


class DummySourceOperator(Operator):
    """
    A dummy source operator that always returns two shards of data.

    This operator is useful for testing purposes.
    """

    def __init__(self, id: str, input_ids: List[str], config: DummySourceOperatorConfig):
        super().__init__(id, input_ids, config)
        self.num_rows = config.num_rows

    def execute(self, inputs: DatasetRefs) -> List[ShardRef]:
        """
        Execute the dummy source operator.

        Args:
            inputs (DatasetRefs): This operator ignores inputs.

        Returns:
            ManyShardRefs: Two shards of dummy data.
        """
        return [
            self.create_dummy_shard.remote(self, 0),
            self.create_dummy_shard.remote(self, 1),
        ]

    @ray.remote
    def create_dummy_shard(self, shard_id: int) -> Dataset:
        """
        Create a dummy shard of data.

        Args:
            shard_id (int): The ID of the shard (0 or 1).

        Returns:
            Dataset: A dummy dataset shard.
        """
        return Dataset.from_dict(
            {
                "id": range(shard_id * self.num_rows, (shard_id + 1) * self.num_rows),
                "output": [f"Sample text {i}" for i in range(self.num_rows)],
            }
        )


def register_dummy_operator():
    register_operator(DummySourceOperatorConfig, DummySourceOperator)
