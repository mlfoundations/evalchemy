from typing import Literal, Optional, Dict, Any, List, Type
from pydantic import BaseModel, Field

class OperatorSpecificConfig(BaseModel):
    type: str
    
class OperatorConfig(BaseModel):
    id: str
    input_ids: List[str] = Field(default_factory=list)
    config: OperatorSpecificConfig
    
    class Config:
        extra = "forbid"

class FunctionOperatorConfig(OperatorSpecificConfig):
    type: Literal["function"] = "function"
    function: str
    function_config: Dict[str, Any] = Field(default_factory=dict)
    sharded: bool = False
    num_shards: int = 3

class HFSourceOperatorConfig(OperatorSpecificConfig):
    type: Literal["hf_source"] = "hf_source"
    dataset: str
    split: str
    columns: Optional[List[str]] = None
    num_truncate: Optional[int] = None

CONFIG_TYPE_MAP: Dict[str, Type[OperatorSpecificConfig]] = {}


def get_config_class(config_type: str) -> Type[OperatorSpecificConfig]:
    return CONFIG_TYPE_MAP.get(config_type)
