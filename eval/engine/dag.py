from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from engine.operators.hashing_utils import HashCodeHelper
from engine.operators.operator import Operator, OperatorSpecificConfig


class DAG(BaseModel):
    """
    Represents the structure of a Directed Acyclic Graph (DAG) of operators.

    Attributes:
        operators (List[Operator]): List of operators in the DAG.
        output_ids (List[str]): List of operator IDs to be used as output.
    """

    operators: List[Operator] = Field(default_factory=list)
    output_ids: List[str] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_operator(self, operator: Operator):
        """Add an operator to the DAG."""
        self.operators.append(operator)

    def pop(self, index: int):
        if index >= len(self.operators):
            raise ValueError("Popped index not in operators")

        self.operators.pop(index)

    def extend(self, another_dag: "DAG"):
        for op in another_dag.operators:
            self.add_operator(op)

    def set_output_ids(self, output_ids: List[str]):
        """Set the output IDs for the DAG."""
        self.output_ids = output_ids

    def get_operator_by_id(self, operator_id: str) -> Operator:
        """Get an operator by its ID."""
        for operator in self.operators:
            if operator.id == operator_id:
                return operator
        raise ValueError(f"Operator with ID {operator_id} not found in the DAG.")

    def get_out_degree_map(self):
        out_degree_map = {}
        for op in self.operators:
            for input_id in op.input_ids:
                out_degree = out_degree_map.get(input_id, 0)
                out_degree_map[input_id] = out_degree + 1
        return out_degree_map

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

    def calculate_operator_hashes(self, sorted_ops: List[Operator], hasher: HashCodeHelper) -> Dict[Operator, str]:
        op_map = {op.id: op for op in self.operators}

        ancestor_configs = self.get_ancestor_configs(op_map, sorted_ops)
        return {op.id: hasher.hash_operator_config_list(ancestor_configs[op.id]) for op in sorted_ops}

    def get_ancestor_operators(
        self, op_map: Dict[str, Operator], sorted_operators: List[Operator]
    ) -> Dict[str, List[Operator]]:
        # Initialize a dictionary to store the ancestor operators for each op
        ancestor_operators = {}

        def get_op_ancestors(op_id: str):
            # If we've already computed this op's ancestors, return them
            if op_id in ancestor_operators:
                return ancestor_operators[op_id]

            op = op_map[op_id]

            # If this op has no parents, its ancestor list contains only itself
            if not op.input_ids:
                ancestor_operators[op_id] = [op]
                return ancestor_operators[op_id]

            # Get ancestors for all parents, preserving order
            parents_ancestors = []
            for parent_id in op.input_ids:
                parents_ancestors.extend(get_op_ancestors(parent_id))

            # Remove duplicates while preserving order
            unique_ancestors = []
            seen = set()
            for ancestor in parents_ancestors:
                if ancestor.id not in seen:
                    unique_ancestors.append(ancestor)
                    seen.add(ancestor.id)

            # Add this op to the end of the unique ancestors
            ancestor_operators[op_id] = unique_ancestors + [op]
            return ancestor_operators[op_id]

        # Iterate through the sorted operators and compute their ancestors
        for op in sorted_operators:
            get_op_ancestors(op.id)

        return ancestor_operators

    def get_ancestor_configs(
        self, op_map: Dict[str, Operator], sorted_operators: List[Operator]
    ) -> Dict[str, OperatorSpecificConfig]:
        ancestor_operators = self.get_ancestor_operators(op_map, sorted_operators)
        ancestor_configs = {op_id: [op.config for op in ops] for op_id, ops in ancestor_operators.items()}
        return ancestor_configs

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the DAG.

        Returns:
            Dict[str, Any]: A dictionary containing the DAG's operators and output_ids.
        """
        return {"operators": [op.to_dict() for op in self.operators], "output_ids": self.output_ids}
