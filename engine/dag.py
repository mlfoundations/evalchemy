from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from engine.operators.operator import (
    Operator,
    OperatorConfig,
    OperatorSpecificConfig,
    create_operator,
    get_config_class,
)


class DAG(BaseModel):
    """
    Represents the structure of a Directed Acyclic Graph (DAG) of operators.

    Attributes:
        name (str): Unique identifier of the DAG.
        operators (List[Operator]): List of operators in the DAG.
        output_ids (List[str]): List of operator IDs to be used as output.
    """

    name: str
    operators: List[Operator] = Field(default_factory=list)
    output_ids: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_operator(self, operator: Operator):
        """Add an operator to the DAG."""
        self.operators.append(operator)

    def set_output_ids(self, output_ids: List[str]):
        """Set the output IDs for the DAG."""
        self.output_ids = output_ids

    def get_operator_by_id(self, operator_id: str) -> Operator:
        """Get an operator by its ID."""
        for operator in self.operators:
            if operator.id == operator_id:
                return operator
        raise ValueError(f"Operator with ID {operator_id} not found in the DAG.")

    def validate(self):
        """Validate the DAG structure."""
        operator_ids = set(op.id for op in self.operators)
        for operator in self.operators:
            for input_id in operator.input_ids:
                if input_id not in operator_ids:
                    raise ValueError(f"Input ID {input_id} for operator {operator.id} not found in the DAG.")
        for output_id in self.output_ids:
            if output_id not in operator_ids:
                raise ValueError(f"Output ID {output_id} not found in the DAG.")

    def topological_sort(self) -> List[Operator]:
        """Perform a topological sort of the operators in the DAG."""
        # Create a copy of the graph
        graph = {op.id: set(op.input_ids) for op in self.operators}
        sorted_ops = []
        no_incoming = [op for op in self.operators if not op.input_ids]

        while no_incoming:
            node = no_incoming.pop(0)
            sorted_ops.append(node)

            for op in self.operators:
                if node.id in graph[op.id]:
                    graph[op.id].remove(node.id)
                    if not graph[op.id]:
                        no_incoming.append(op)

        if len(sorted_ops) != len(self.operators):
            raise ValueError("The DAG contains a cycle")

        return sorted_ops


def parse_dag(config: Dict, sub_dir: Optional[Tuple[str, ...]] = None) -> DAG:
    """
    Parse the configuration and create a DAG.

    Args:
        config (Dict): The configuration dictionary.
        sub_dir (Optional[Tuple[str, ...]]): Subdirectory within the config to use.

    Returns:
        DAG: The created DAG.

    Raises:
        ValueError: If there are duplicate operator IDs or invalid configurations.
    """
    if sub_dir is not None:
        for key in sub_dir:
            config = config[key]
    dag = DAG(name=config["name"])

    seen_ids = set()
    previous_op_id = None
    for op_config in config["operators"]:
        op_id = op_config["id"]
        if op_id in seen_ids:
            raise ValueError(f"Duplicate operator ID found: {op_id}")
        seen_ids.add(op_id)

        # If input_ids is not specified, use the previous operator's ID
        if "input_ids" not in op_config and previous_op_id is not None:
            op_config["input_ids"] = [previous_op_id]

        try:
            specific_config = parse_specific_config(op_config["config"])
            operator_config = OperatorConfig(id=op_id, input_ids=op_config.get("input_ids", []), config=specific_config)
            operator = create_operator(operator_config)
            dag.add_operator(operator)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration for operator {op_id}: {str(e)}")

        previous_op_id = op_id

    # If output_ids is not specified, use the last operator's ID
    if "output_ids" not in config:
        if dag.operators:
            dag.set_output_ids([dag.operators[-1].id])
    else:
        dag.set_output_ids(config["output_ids"])

    try:
        dag.validate()
    except ValueError as e:
        raise ValueError(f"Invalid DAG structure: {str(e)}")

    return dag


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


def load_dag(config_path: str, sub_dir: Optional[Tuple[str, ...]] = None) -> DAG:
    """
    Load a DAG from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
        sub_dir (Optional[Tuple[str, ...]]): Subdirectory within the config to use.

    Returns:
        DAG: The loaded DAG.
    """
    config = parse_yaml_config(config_path)
    return parse_dag(config, sub_dir)
