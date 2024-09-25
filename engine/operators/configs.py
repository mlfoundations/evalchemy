from typing import Literal, Optional, Dict, Any, List, Type
from pydantic import BaseModel, Field


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


CONFIG_TYPE_MAP: Dict[str, Type[OperatorSpecificConfig]] = {}


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
