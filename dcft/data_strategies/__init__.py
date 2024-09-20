import os

import yaml
import sys
import pandas as pd
from datasets import Dataset
from typing import Dict, Any, List, Optional

from dcft.data_strategies.dataset_utils import DatasetHandler, SyntheticDataFramework, check_dataset_mix_in_yaml

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class SyntheticDataManager:
    def __init__(self):
        self.strategies_dir = os.path.dirname(os.path.abspath(__file__))
        self.frameworks = self._load_frameworks()

    def _load_frameworks(self) -> Dict[str, SyntheticDataFramework]:
        frameworks = {}
        for strategy_dir in os.listdir(self.strategies_dir):
            strategy_path = os.path.join(self.strategies_dir, strategy_dir)
            if os.path.isdir(strategy_path) and strategy_dir != "__pycache__":
                for file in os.listdir(strategy_path):
                    if file.endswith(".yaml"):
                        config_path = os.path.join(strategy_path, file)
                        if check_dataset_mix_in_yaml(config_path):
                            framework = DatasetHandler.from_config(config_path)
                        else:
                            framework = SyntheticDataFramework.from_config(config_path)
                        
                        if framework.name in frameworks:
                            raise ValueError(f"Invalid name: {framework.name} is duplicated")
                        frameworks[framework.name] = framework
        return frameworks

    def get_framework(self, framework_name: str) -> SyntheticDataFramework:
        return self.frameworks.get(framework_name)

    def list_frameworks(self) -> List[str]:
        return list(self.frameworks.keys())

    def run_framework(self, framework_name: str) -> None:
        framework = self.get_framework(framework_name)
        if framework:
            print(f"Running framework: {framework_name}")
            framework.run()
        else:
            print(f"Framework '{framework_name}' not found.")
        
        framework.generated_dataset.push_to_hub(f"EtashGuha/{framework.name}")

# Example usage
if __name__ == "__main__":
    manager = SyntheticDataManager()
    print("Available frameworks:", manager.list_frameworks())

    # Run a specific framework
    framework_name = "example_framework"
    manager.run_framework(framework_name)
