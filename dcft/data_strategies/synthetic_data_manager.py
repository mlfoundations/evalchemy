import os
from typing import Dict, List

from engine.dag import DAG, parse_dag


class SyntheticDataManager:
    """
    A manager class for handling data frameworks or dataset handlers, outside of the data generation process itself.

    This class is responsible for loading, managing, and running various
    synthetic data generation frameworks based on YAML configurations.
    """

    def __init__(self):
        """
        Initialize the SyntheticDataManager.

        Sets up the strategies directory and loads all available frameworks.
        """
        self.strategies_dir = os.path.dirname(os.path.abspath(__file__))
        self.frameworks = self._load_frameworks()

    def list_frameworks(self) -> List[str]:
        """
        List all available framework names.

        Returns:
            List[str]: A list of all loaded framework names.
        """
        return list(self.frameworks.keys())

    def run_framework(self, framework_name: str, hf_account: str) -> None:
        """
        Run a specific framework and push the generated dataset to Hugging Face Hub.

        Args:
            framework_name (str): The name of the framework to run.
            hf_account (str): The Hugging Face account name to push the dataset to.

        Note:
            If the framework is a DatasetHandler, it will be provided with all loaded frameworks.
        """
        framework = self.get_framework(framework_name)

        if framework:
            print(f"Running framework: {framework_name}")
            framework.run()
        else:
            print(f"Framework '{framework_name}' not found.")

        framework.generated_dataset.push_to_hub(f"{hf_account}/{framework.name}")

    def _load_frameworks(self) -> Dict[str, DAG]:
        """
        Load all synthetic data frameworks from YAML configurations.

        This method scans the strategies directory for YAML files and loads
        them as either DatasetHandler or SyntheticDataFramework instances.

        Returns:
            Dict[str, SyntheticDataFramework]: A dictionary mapping framework names to their instances.

        Raises:
            ValueError: If a duplicate framework name is encountered.
        """
        frameworks = {}
        for strategy_dir in os.listdir(self.strategies_dir):
            strategy_path = os.path.join(self.strategies_dir, strategy_dir)
            if os.path.isdir(strategy_path) and strategy_dir != "__pycache__":
                for file in os.listdir(strategy_path):
                    if file.endswith(".yaml"):
                        config_path = os.path.join(strategy_path, file)
                        dag = parse_dag(config_path)
                        frameworks[dag.name] = dag
        return frameworks
