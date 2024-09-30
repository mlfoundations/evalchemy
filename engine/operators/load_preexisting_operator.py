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
        self.dag = self._load_frameworks(config.strategies_dir, config.framework_name)

    def execute(self, _: DatasetRefs) -> ManyShardRefs:
        """
        Execute the load preexisting operator.

        Args:
            inputs (DatasetRefs): Map of input datasets (not used in this operator)

        Returns:
            ManyShardRefs: List of waitables (shards) outputted by the executed framework
        """
        from engine.executor import DAGExecutor
        return DAGExecutor(self.dag).get_waitables()

    def _load_frameworks(self, strategies_dir: str, framework_name: str):
        from engine.dag import load_dag  
        for strategy_dir in os.listdir(strategies_dir):
            strategy_path = os.path.join(strategies_dir, strategy_dir)
            if os.path.isdir(strategy_path) and strategy_dir != "__pycache__":
                for file in os.listdir(strategy_path):
                    if file.endswith(".yaml"):
                        config_path = os.path.join(strategy_path, file)
                        dag = load_dag(config_path)
                        if dag.name == framework_name:
                            return dag
        raise ValueError(f"Framework '{framework_name}' not found in {strategies_dir}.")
    
register_operator(LoadPreexistingOperatorConfig, LoadPreexistingOperator)
