from typing import Literal, Optional, Dict, Any, List, Type, Callable
from pydantic import BaseModel, Field
import inspect
import hashlib
import base64
import ast

from engine.operators.hashing_utils import hash_function


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
    num_shards: int = 1


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


import ast
import inspect
import hashlib
import os


def get_local_imports_and_calls(node, base_path):
    local_imports = []
    function_calls = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Import):
            for alias in item.names:
                if not is_standard_library(alias.name):
                    local_imports.append(alias.name)
        elif isinstance(item, ast.ImportFrom):
            if item.level > 0 or not is_standard_library(item.module):
                local_imports.append(item.module)
        elif isinstance(item, ast.Call):
            if isinstance(item.func, ast.Name):
                function_calls.add(item.func.id)
            elif isinstance(item.func, ast.Attribute):
                function_calls.add(item.func.attr)
    return [os.path.join(base_path, imp.replace(".", "/") + ".py") for imp in local_imports], function_calls


def is_standard_library(module_name):
    try:
        module = __import__(module_name)
        return hasattr(module, "__file__") and module.__file__.startswith(os.path.dirname(os.__file__))
    except ImportError:
        return False


def get_file_content(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_function_source(func_name, tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(tree.body[0].source, node)
    return None


def hash_function_with_imports_and_calls(func):
    # Get the source code of the function
    source = inspect.getsource(func)

    # Parse the source code
    tree = ast.parse(source)

    # Get the base path (assuming the function is in a file)
    base_path = os.path.dirname(inspect.getfile(func))

    # Get local imports and function calls
    local_imports, function_calls = get_local_imports_and_calls(tree, base_path)
    breakpoint()

    # Recursively get content of local imports and called functions
    def get_recursive_content(file_path, visited=None, calls=None):
        if visited is None:
            visited = set()
        if calls is None:
            calls = set()

        if file_path in visited:
            return ""

        visited.add(file_path)
        content = get_file_content(file_path)
        tree = ast.parse(content)
        imports, new_calls = get_local_imports_and_calls(tree, os.path.dirname(file_path))

        # Add sources of called functions
        for call in calls:
            func_source = get_function_source(call, tree)
            if func_source:
                content += "\n" + func_source

        return content + "".join(get_recursive_content(imp, visited, new_calls) for imp in imports)

    # Concatenate all code
    all_code = source + "".join(get_recursive_content(imp, set(), function_calls) for imp in local_imports)
    print(all_code)
    # Generate hash
    return hashlib.sha256(all_code.encode()).hexdigest()


def hash_value(value: Any) -> str:
    """
    Create a hash string for a value, handling unhashable types.

    Args:
        value (Any): The value to hash.

    Returns:
        str: A hash string value.
    """
    if isinstance(value, Callable):
        # For functions, hash the function name and its source code
        return hash_function(value)
    elif isinstance(value, (list, tuple)):
        # For lists and tuples, hash their contents recursively
        return f"list:{','.join(hash_value(item) for item in value)}"
    elif isinstance(value, dict):
        # For dictionaries, hash their items recursively
        return f"dict:{','.join(f'{k}:{hash_value(v)}' for k, v in sorted(value.items()))}"
    elif isinstance(value, set):
        # For sets, convert to a sorted list and hash
        return f"set:{','.join(sorted(hash_value(item) for item in value))}"
    elif hasattr(value, "__dict__"):
        # For objects, hash their __dict__
        return f"obj:{hash_value(value.__dict__)}"
    else:
        # For other types, convert to string and hash
        return f"val:{hashlib.sha256(str(value).encode()).hexdigest()[:16]}"


def hash_operator_config_list(config_list: List[OperatorSpecificConfig]) -> str:
    """
    Hash a list of OperatorSpecificConfig objects to a sanitized string.

    Args:
        config_list (List[OperatorSpecificConfig]): List of OperatorSpecificConfig objects to hash.

    Returns:
        str: Sanitized string hash value of the list.
    """
    # Convert each config to a string of "key:hashed_value" pairs
    config_strings = [",".join(f"{k}:{hash_value(v)}" for k, v in config.dict().items()) for config in config_list]

    # Join all config strings and create a final hash
    full_string = "|".join(config_strings)
    hash_object = hashlib.sha256(full_string.encode())
    # Convert to URL-safe base64 and remove padding
    return base64.urlsafe_b64encode(hash_object.digest()).decode().rstrip("=")
