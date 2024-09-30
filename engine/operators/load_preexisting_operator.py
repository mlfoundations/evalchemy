import os
from typing import List, Literal

import ray
from pydantic import Field

from engine.operators.operator import (
    DatasetRefs,
    ManyShardRefs,
    Operator,
    OperatorSpecificConfig,
    register_operator,
)

class LoadPreexistingOperatorConfig(OperatorSpecificConfig):
    """
    Configuration class for load preexisting operators.

    Attributes:
        type (Literal["load_preexisting"]): The type of the operator, always set to "load_preexisting".
        framework_name (str): The name of the framework to load and execute.
    """
    type: Literal["load_preexisting"] = "load_preexisting"
    framework_name: str
    strategies_dir: str = Field(default="dcft/data_strategies")

class LoadPreexistingOperator(Operator):
    """
    Operator that loads and executes a preexisting framework.

    Attributes:
        framework_name (str): The name of the framework to load and execute.
    """

    def __init__(self, id: str, input_ids: List[str], config: LoadPreexistingOperatorConfig):
        """
        Initialize the LoadPreexistingOperator.

        Args:
            id (str): Unique identifier for the operator.
            input_ids (List[str]): List of input identifiers for the operator.
            config (LoadPreexistingOperatorConfig): Specific configuration for the load preexisting operator.
        """
        super().__init__(id, input_ids, config)
        self.framework_name = config.framework_name
        self.strategies_dir = config.strategies_dir

    def execute(self, _: DatasetRefs) -> ManyShardRefs:
        """
        Execute the load preexisting operator.

        Args:
            inputs (DatasetRefs): Map of input datasets (not used in this operator)

        Returns:
            ManyShardRefs: List of waitables (shards) outputted by the executed framework
        """
        return self.load_and_execute_framework.remote(self)

    def _load_frameworks(self, strategies_dir: str, framework_name: str):
        from engine.dag import parse_dag  # Import here to avoid circular import
        for file in os.listdir(strategies_dir):
            if file.endswith(".yaml"):
                config_path = os.path.join(strategies_dir, file)
                dag = parse_dag(config_path)
                if dag.name == framework_name:
                    return dag
        raise ValueError(f"Framework '{framework_name}' not found in {strategies_dir}.")
    
    @ray.remote
    def load_and_execute_framework(self):
        from engine.executor import DAGExecutor
        dag = self._load_frameworks(self.strategies_dir, self.framework_name)
        executor = DAGExecutor(dag)
        return executor.execute()

register_operator(LoadPreexistingOperatorConfig, LoadPreexistingOperator)
