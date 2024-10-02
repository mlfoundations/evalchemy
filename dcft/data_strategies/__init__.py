import os
import sys
from typing import Dict, List
import fsspec

from dcft.data_strategies.dataset_utils import DatasetHandler, SyntheticDataFramework
from dcft.data_strategies.yaml_utils import check_dataset_mix_in_yaml

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class SyntheticDataManager:
    """
    A manager class for handling data frameworks or dataset handlers, outside of the data generation process itself.

    This class is responsible for loading, managing, and running various
    synthetic data generation frameworks based on YAML configurations.
    """

    def __init__(self, hf_account: str, cache_dir: str, fs_type: str = "file", overwrite_cache: bool = False) -> None:
        """
        Initialize the SyntheticDataManager.

        Sets up the strategies directory and loads all available frameworks.
        """
        self.hf_account = hf_account
        self.cache_dir = cache_dir
        self.fs_type = fs_type
        self.overwrite_cache = overwrite_cache
        self.strategies_dir = os.path.dirname(os.path.abspath(__file__))
        self.frameworks = self._load_frameworks()

    def _load_frameworks(self) -> Dict[str, SyntheticDataFramework]:
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
                        try:
                            if check_dataset_mix_in_yaml(config_path):
                                framework = DatasetHandler.from_config(
                                    config_path,
                                    cache_dir=self.cache_dir,
                                    fs=fsspec.filesystem(self.fs_type),
                                    overwrite_cache=self.overwrite_cache,
                                )
                            else:
                                framework = SyntheticDataFramework.from_config(
                                    config_path,
                                    cache_dir=self.cache_dir,
                                    fs=fsspec.filesystem(self.fs_type),
                                    overwrite_cache=self.overwrite_cache,
                                )
                        except:
                            print(f"Could not load from {config_path}")
                            continue
                        if framework.name in frameworks:
                            raise ValueError(f"Invalid name: {framework.name} is duplicated")
                        frameworks[framework.name] = framework
        return frameworks

    def get_framework(self, framework_name: str) -> SyntheticDataFramework:
        """
        Retrieve a specific framework by name.

        Args:
            framework_name (str): The name of the framework to retrieve.

        Returns:
            SyntheticDataFramework: The requested framework instance, or None if not found.
        """
        return self.frameworks.get(framework_name)

    def list_frameworks(self) -> List[str]:
        """
        List all available framework names.

        Returns:
            List[str]: A list of all loaded framework names.
        """
        return list(self.frameworks.keys())

    def run_framework(self, framework_name: str) -> None:
        """
        Run a specific framework and push the generated dataset to Hugging Face Hub.

        Args:
            framework_name (str): The name of the framework to run.
            hf_account (str): The Hugging Face account name to push the dataset to.

        Note:
            If the framework is a DatasetHandler, it will be provided with all loaded frameworks.
        """
        framework = self.get_framework(framework_name)

        if isinstance(framework, DatasetHandler):
            framework.all_loaded_frameworks = self.frameworks

        if framework:
            print(f"Running framework: {framework_name}")
            framework.run()
        else:
            print(f"Framework '{framework_name}' not found.")

        framework.generated_dataset.push_to_hub(f"{self.hf_account}/{framework.name}")
