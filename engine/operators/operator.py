from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field

from engine.operators.configs import OperatorSpecificConfig
from engine.dataset import DatasetRef, DatasetRefs


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
    def execute(self, inputs: DatasetRefs) -> DatasetRef:
        pass


class OperatorConfig(BaseModel):
    id: str
    input_ids: List[str] = Field(default_factory=list)
    config: OperatorSpecificConfig
    
    class Config:
        extra = "forbid"


def create_operator(config: OperatorConfig) -> Operator:
    from engine.operators.registry import get_operator_class

    operator_class = get_operator_class(type(config.config))
    if operator_class is None:
        raise ValueError(f"Unknown operator type: {type(config.config)}")
    return operator_class(config.id, config.input_ids, config.config)
