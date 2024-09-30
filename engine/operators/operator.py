from abc import ABC, abstractmethod
from typing import Dict, List, Type, TypeAlias

import ray
from pydantic import BaseModel, Field

ShardRef: TypeAlias = ray.ObjectRef
ManyShardRefs: TypeAlias = List[ShardRef]
DatasetRefs: TypeAlias = Dict[str, ManyShardRefs]


class OperatorSpecificConfig(BaseModel):
    """
    Base class for operator-specific configurations.

    Attributes:
        type (str): The type of the operator.
    """

    type: str


class OperatorConfig(BaseModel):
    """
    Configuration class for operators.

    Attributes:
        id (str): Unique identifier for the operator.
        input_ids (List[str]): List of input identifiers for the operator.
        config (OperatorSpecificConfig): Specific configuration for the operator.

    Config:
        extra (str): Set to "forbid" to disallow extra attributes.
    """

    id: str
    input_ids: List[str] = Field(default_factory=list)
    config: OperatorSpecificConfig

    class Config:
        extra = "forbid"


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

    @abstractmethod
    def execute(self, inputs: DatasetRefs) -> ManyShardRefs:
        """
        Execute the operator on the given inputs.

        Args:
            inputs (DatasetRefs): Dictionary of inputs mapping identifiers to a list of shard references (known as a dataset)

        Returns:
            ManyShardRefs: List of processed output shard references for each input shard
        """
        pass


OPERATOR_MAP: Dict[Type[OperatorSpecificConfig], Type[Operator]] = {}
CONFIG_TYPE_MAP: Dict[str, Type[OperatorSpecificConfig]] = {}


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


def get_config_class(config_type: str) -> Type[OperatorSpecificConfig]:
    """
    Get the configuration class for a given operator type.

    Args:
        config_type (str): The type of the operator configuration.

    Returns:
        Type[OperatorSpecificConfig]: The corresponding configuration class for the given type.
        If the type is not found in the CONFIG_TYPE_MAP, returns None.
    """
    return CONFIG_TYPE_MAP.get(config_type)
