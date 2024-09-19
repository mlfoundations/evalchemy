import os
import yaml
import importlib
import yaml
import sys
import pandas as pd
from datasets import Dataset
from typing import Dict, Any, List, Optional

from dcft.data_strategies.huggingface_utils import HuggingFaceUploader
from dcft.data_strategies.dataset_generation import (
    InstructionGenerator,
    InstructionFilter,
    AnnotationGenerator,
    ModelPairFilter,
    AnnotationSeeder,
    InstructionSeeder,
)
from dcft.data_strategies.dataset_utils import DatasetMixer, DatasetShuffler, DatasetCache, DatasetSaver

from lm_eval.utils import eval_logger

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class SyntheticDataFramework:
    """
    Main class for orchestrating the synthetic data generation process.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        instruction_generator: Optional[InstructionGenerator] = None,
        instruction_filter: Optional[InstructionFilter] = None,
        annotation_generator: Optional[AnnotationGenerator] = None,
        model_pair_filter: Optional[ModelPairFilter] = None,
        dataset_mixer: Optional[DatasetMixer] = None,
        dataset_shuffler: Optional[DatasetShuffler] = None,
        dataset_cache: Optional[DatasetCache] = None,
        dataset_saver: Optional[DatasetSaver] = None,
        huggingface_uploader: Optional[HuggingFaceUploader] = None,
    ):
        """
        Initialize the framework with optional components.

        All parameters default to None if not provided.
        """
        self.name = name
        self.config = config
        self.instruction_generator = instruction_generator
        self.instruction_filter = instruction_filter
        self.annotation_generator = annotation_generator
        self.model_pair_filter = model_pair_filter
        self.dataset_mixer = dataset_mixer
        self.dataset_shuffler = dataset_shuffler
        self.dataset_cache = dataset_cache
        self.dataset_saver = dataset_saver
        self.huggingface_uploader = huggingface_uploader

    @staticmethod
    def from_config(config_path: str) -> "SyntheticDataFramework":
        """
        Create and return a SyntheticDataFramework instance from a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            SyntheticDataFramework: An instance of the class with initialized components.
        """
        framework = SyntheticDataFramework()
        framework._load_config(config_path)
        return framework

    def _load_config(self, yaml_path: str) -> Dict[str, Any]:
        """
        Load the configuration and initialize components.

        Args:
            config_path (str): Path to the YAML configuration file.
        """

        def function_constructor(loader, node):
            value = loader.construct_scalar(node)
            try:
                func_name, args = value.split(":", 1)
                args = args.strip()
            except ValueError:
                func_name, args = value, ""

            strategy_dir, func_name = func_name.rsplit(".", 1)
            utils_module = self._import_utils_module(strategy_dir)

            func = getattr(utils_module, func_name)
            return func

        # Add the custom constructor for the !function tag
        yaml.add_constructor("!function", function_constructor, Loader=yaml.SafeLoader)

        # Load the YAML file with the custom constructor
        with open(yaml_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.name = self.config["name"]

        self.instruction_generator = InstructionGenerator(self.config["instruction_generation"])
        self.instruction_filter = InstructionFilter(self.config["instruction_filtering"])
        self.annotation_generator = AnnotationGenerator(self.config["annotation_generation"])
        self.model_pair_filter = ModelPairFilter(self.config["model_pair_filtering"])
        self.annotation_seeder = AnnotationSeeder(self.config["annotation_seeder"])
        self.instruction_seeder = InstructionSeeder(self.config["instruction_seeder"])

    def generate_dataset(self) -> None:
        """
        Execute the complete pipeline for dataset generation.
        """
        if not self.config:
            raise ValueError("Configuration not loaded. Use from_config() to create an instance.")

        eval_logger.info("Seeding Instructions")
        instruction_seeds = self.instruction_seeder.generate()
        eval_logger.info("Generating Instructions")
        instructions = self.instruction_generator.generate(instruction_seeds)
        eval_logger.info("Filtering Instructions")
        filtered_instructions = self.instruction_filter.filter(instructions)
        eval_logger.info("Annotating Generations")
        annotations = self.annotation_generator.generate(filtered_instructions)
        eval_logger.info("Filtering Paris")
        filtered_pairs = self.model_pair_filter.filter(annotations)
        self._upload_to_huggingface(filtered_pairs)

    def _upload_to_huggingface(self, data_pairs):
        df = pd.DataFrame(data_pairs)

        # Convert pandas DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)

        dataset.push_to_hub(f"EtashGuha/{self.name}")

    def _import_utils_module(self, strategy_dir: str):
        module_name = f"dcft.data_strategies.{strategy_dir}"
        return importlib.import_module(module_name)

    def run(self) -> None:
        """
        Run the entire data generation process.
        """
        self.generate_dataset()


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


# Example usage
if __name__ == "__main__":
    manager = SyntheticDataManager()
    print("Available frameworks:", manager.list_frameworks())

    # Run a specific framework
    framework_name = "example_framework"
    manager.run_framework(framework_name)
