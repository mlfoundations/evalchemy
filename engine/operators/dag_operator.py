from typing import Dict, List, Literal

from pydantic import Field

from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefs,
    Operator,
    OperatorSpecificConfig,
    register_operator,
)


class DAGOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for DAG operators.

    Attributes:
        type (Literal["dag"]): The type of the operator, always set to "dag".
        dag (Dict): The DAG configuration to be executed.
    """
    type: Literal["dag"] = "dag"
    dag: Dict = Field(...)


class DAGOperator(Operator):
    """
    Operator that executes a DAG specified in its configuration.

    Attributes:
        dag (DAG): The DAG to be executed.
    """

    def __init__(self, id: str, input_ids: List[str], config: DAGOperatorConfig):
        """
        Initialize the DAGOperator.

        Args:
            id (str): Unique identifier for the operator.
            input_ids (List[str]): List of input identifiers for the operator.
            config (DAGOperatorConfig): Specific configuration for the DAG operator.
        """
        from engine.dag import parse_dag
        super().__init__(id, input_ids, config)
        self.dag = parse_dag(config.dag)

    def execute(self, inputs: DatasetRefs) -> ManyShardRefs:
        """
        Execute the DAG operator.

        Args:
            inputs (DatasetRefs): Map of input datasets

        Returns:
            ManyShardRefs: List of waitables (shards) outputted by the executed DAG
        """
        from engine.executor import DAGExecutor
        executor = DAGExecutor(self.dag)
        return executor.get_waitables()


# Register the operator
register_operator(DAGOperatorConfig, DAGOperator)