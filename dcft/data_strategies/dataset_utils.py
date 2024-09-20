from abc import ABC, abstractmethod
import yaml
import importlib
from typing import List, Dict, Any
from datasets import Dataset, concatenate_datasets
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dcft.data_strategies.dataset_generation import (
    InstructionGenerator,
    InstructionFilter,
    AnnotationGenerator,
    ModelPairFilter,
    AnnotationSeeder,
    InstructionSeeder,
)
from typing import Dict, Any, List, Optional
from lm_eval.utils import eval_logger
import pandas as pd

class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None

def check_dataset_mix_in_yaml(file_path):
    # Add a constructor that ignores unknown tags
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    try:
        with open(file_path, 'r') as f:
            config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
        return 'dataset_mix' in config if isinstance(config, dict) else False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return False
    except IOError as e:
        print(f"Error opening file {file_path}: {e}")
        return False
    

def _get_len_subcomponents(file_path):
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(file_path, 'r') as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
    
    return len(config['dataset_mix'])

def _get_name(file_path):
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(file_path, 'r') as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
    
    return config['name']

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
        self.generated_dataset = None

    @staticmethod
    def from_config(config_path: str, sub_dir: Optional[tuple] = None) -> "SyntheticDataFramework":
        """
        Create and return a SyntheticDataFramework instance from a configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
            sub_dir Optional(str): Directory in yaml where config is located

        Returns:
            SyntheticDataFramework: An instance of the class with initialized components.
        """
        framework = SyntheticDataFramework()
        framework._load_config(config_path, sub_dir)
        return framework

    def _load_config(self, yaml_path: str, sub_dir: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Load the configuration and initialize components.

        Args:
            config_path (str): Path to the YAML configuration file.
            sub_dir Optional(str): Directory in yaml where config is located
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
        if sub_dir is not None:
            for key in sub_dir:
                self.config = self.config[key]

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

        df = pd.DataFrame(filtered_pairs)

        # Convert pandas DataFrame to Hugging Face Dataset
        self.generated_dataset = Dataset.from_pandas(df)

    def _import_utils_module(self, strategy_dir: str):
        module_name = f"dcft.data_strategies.{strategy_dir}"
        return importlib.import_module(module_name)

    def run(self) -> None:
        """
        Run the entire data generation process.
        """
        self.generate_dataset()



class DatasetHandler:
    def __init__(self, sub_frameworks_lazy: List):
        self.all_sub_frameworks_lazy = sub_frameworks_lazy
        self.all_sub_frameworks = None
        self.max_workers = os.cpu_count()
        self.shuffle_seed = 42

    @staticmethod
    def from_config(config_path: str) -> "DatasetHandler":
        num_components = _get_len_subcomponents(config_path)
        all_sub_frameworks_lazy = []
        for index in range(num_components):
            all_sub_frameworks_lazy.append((config_path, ("dataset_mix", index)))
        dataset_handler = DatasetHandler(all_sub_frameworks_lazy)
        dataset_handler.name = _get_name(config_path)
        return dataset_handler

    def mix(self, datasets: List[Dataset]) -> Dataset:
        print("Starting dataset mixing process...")

        # Combine datasets
        combined_dataset = self._combine_datasets(datasets)
        print(f"Combined {len(datasets)} datasets, total items: {len(combined_dataset)}")
        
        # Shuffle
        shuffled_dataset = self.shuffle(combined_dataset)
        print("Dataset shuffled")
        
        return shuffled_dataset

    def _combine_datasets(self, datasets: List[Dataset]) -> Dataset:
        return concatenate_datasets(datasets)

    def shuffle(self, dataset: Dataset) -> Dataset:
        return dataset.shuffle(seed=self.shuffle_seed)

    def process_datasets_parallel(self, dataset_configs: List) -> List[Dataset]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_config = {executor.submit(self._load_dataset, config): config for config in dataset_configs}
            results = {}
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    data = future.result()
                    results[data.name] = data
                except Exception as exc:
                    print(f"Dataset {config.get('name', 'unknown')} generated an exception: {exc}")
        return results

    def _load_dataset(self, dataset_args) -> SyntheticDataFramework:
        config_path = dataset_args[0]
        sub_dir = dataset_args[1]
        
        framework = SyntheticDataFramework.from_config(config_path, sub_dir)
        framework.generate_dataset()
        
        return framework
    
    def run(self) -> None:
        all_frameworks = self.process_datasets_parallel(self.all_sub_frameworks_lazy)
        shuffled_datasets = [framework.generated_dataset for framework in all_frameworks.values()]
        self.generated_dataset = self.mix(shuffled_datasets)