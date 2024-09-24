from typing import Literal, Optional, Dict, Any, List
from dataclasses import dataclass
from pydantic import BaseModel, DirectoryPath, Field, HttpUrl

class OperatorSpecificConfig(BaseModel):
    type: str
    
class FunctionOperatorConfig(OperatorSpecificConfig):
    type: Literal["function"] = "function"
    function: str
    function_config: Dict[str, Any] = Field(default_factory=dict)
    sharded: bool = False
    num_shards: int = 20 

class HFSourceOperatorConfig(OperatorSpecificConfig):
    type: Literal["hf_source"] = "hf_source"
    dataset: str
    split: str
    columns: Optional[List[str]] = None
    num_truncate: Optional[int] = None
