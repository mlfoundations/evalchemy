import types
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Type, TypeAlias

import ray
import yaml
from pydantic import BaseModel, Field

ShardRef: TypeAlias = ray.ObjectRef
ManyShardRefsGenerator: TypeAlias = Generator[ShardRef, None, None]
DatasetRefs: TypeAlias = Dict[str, ManyShardRefsGenerator]


class OperatorSpecificConfig(BaseModel):
    """
    Base class for operator-specific configurations.

    Attributes:
        type (str): The type of the operator.
        materialize_output (bool): Whether to materialize the output of the operator.
    """

    type: str
    materialize_output: bool = True


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

    def set_input_ids(self, inp: List[str]) -> None:
        self._input_ids = inp

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

    def execute(self, inputs: DatasetRefs) -> ManyShardRefsGenerator:
        self.outputs = self.compute(inputs)
        return self.outputs

    @abstractmethod
    def compute(self, inputs: DatasetRefs) -> ManyShardRefsGenerator:
        """
        compute the operator on the given inputs.

        Args:
            inputs (DatasetRefs): Dictionary of inputs mapping identifiers to a list of shard references (known as a dataset)

        Returns:
            ManyShardRefsGenerator: A generator of processed output shard references
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the operator.

        Returns:
            Dict[str, Any]: A dictionary containing the operator's id, input_ids, and config.
        """
        return {"id": self.id, "input_ids": self.input_ids, "config": self.config.model_dump()}


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


def parse_specific_config(config: Dict) -> OperatorSpecificConfig:
    """
    Parse the specific configuration for an operator.

    Args:
        config (Dict): The specific configuration dictionary.

    Returns:
        OperatorSpecificConfig: The parsed specific configuration.

    Raises:
        ValueError: If the operator type is unknown.
    """
    operator_type = config["type"]
    config_class = get_config_class(operator_type)

    if config_class is None:
        raise ValueError(f"Unknown operator type: {operator_type}")
    return config_class(**config)


def parse_yaml_config(config_path: str) -> Dict:
    """
    Parse a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict: The parsed configuration as a dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
