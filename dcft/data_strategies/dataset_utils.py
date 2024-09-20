import yaml
import importlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from datasets import Dataset, concatenate_datasets
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from lm_eval.utils import eval_logger
import pandas as pd
from datasets import load_dataset

class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node: Any) -> None:
        return None


def check_dataset_mix_in_yaml(file_path: str) -> bool:
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    try:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
        return "dataset_mix" in config if isinstance(config, dict) else False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return False
    except IOError as e:
        print(f"Error opening file {file_path}: {e}")
        return False


def _get_len_subcomponents(file_path: str) -> int:
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    return len(config["dataset_mix"])


def _get_name(file_path: str, sub_dir: Optional[List[str]] = None) -> str:
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    if sub_dir is not None:
        for key in sub_dir:
            config = config[key]

    return config["name"]


def _get_empty_def(file_path: str, subdir: List[str]) -> bool:
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    for key in subdir:
        config = config[key]

    if len(config.keys()) == 1:
        return True
    else:
        return False


class SyntheticDataFramework:
    def __init__(
        self,
        functions: List[Callable] = None,
        name: Optional[str] = None,
    ):
        self.name = name
        self.functions = functions

    @staticmethod
    def from_config(config_path: str, sub_dir: Optional[Tuple[str, ...]] = None) -> "SyntheticDataFramework":
        framework = SyntheticDataFramework()
        framework._load_config(config_path, sub_dir)
        return framework

    def _load_config(self, yaml_path: str, sub_dir: Optional[Tuple[str, ...]] = None) -> None:
        def function_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
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

        yaml.add_constructor("!function", function_constructor, Loader=yaml.SafeLoader)

        with open(yaml_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        if sub_dir is not None:
            for key in sub_dir:
                self.config = self.config[key]

        self.name = self.config["name"]
        self.functions = self.config['functions']
        
    def generate_dataset(self) -> None:
        if not self.config:
            raise ValueError("Configuration not loaded. Use from_config() to create an instance.")

        eval_logger.info("Seeding Instructions")
        if isinstance(self.functions[0], str):
            split_string = self.functions[0].split("//")
            dataset_name, split_name, seed_name = split_string[0], split_string[1], split_string[2]
            dataset = load_dataset(dataset_name)
            output = dataset[split_name][seed_name]
        elif isinstance(self.functions[0], Callable):
            output = self.functions[0]()

        for function in self.functions[1:]:
            output = function(output)

        filtered_pairs = output

        df = pd.DataFrame(filtered_pairs)

        self.generated_dataset = Dataset.from_pandas(df)

    def _import_utils_module(self, strategy_dir: str) -> Any:
        module_name = f"dcft.data_strategies.{strategy_dir}"
        return importlib.import_module(module_name)

    def run(self) -> None:
        self.generate_dataset()


class DatasetHandler:
    def __init__(self, sub_frameworks_lazy: List[Tuple[str, Tuple[str, int]]]):
        self.all_sub_frameworks_lazy = sub_frameworks_lazy
        self.all_sub_frameworks: Optional[Dict[str, SyntheticDataFramework]] = None
        self.max_workers = os.cpu_count()
        self.shuffle_seed = 42
        self.name: Optional[str] = None
        self.generated_dataset: Optional[Dataset] = None
        self.all_loaded_frameworks: Dict[str, SyntheticDataFramework] = {}

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

        combined_dataset = self._combine_datasets(datasets)
        print(f"Combined {len(datasets)} datasets, total items: {len(combined_dataset)}")

        shuffled_dataset = self.shuffle(combined_dataset)
        print("Dataset shuffled")

        return shuffled_dataset

    def _combine_datasets(self, datasets: List[Dataset]) -> Dataset:
        return concatenate_datasets(datasets)

    def shuffle(self, dataset: Dataset) -> Dataset:
        return dataset.shuffle(seed=self.shuffle_seed)

    def process_datasets_parallel(
        self, dataset_configs: List[Tuple[str, Tuple[str, int]]]
    ) -> Dict[str, SyntheticDataFramework]:
        all_frameworks = []
        for dataset_args in dataset_configs:
            config_path = dataset_args[0]
            sub_dir = dataset_args[1]

            is_empty = _get_empty_def(config_path, sub_dir)
            if is_empty:
                framework_name = _get_name(config_path, sub_dir)
                if framework_name not in self.all_loaded_frameworks:
                    raise ValueError(f"Framework {framework_name} not defined nor loaded in Dataset Mix.")
                framework = self.all_loaded_frameworks[_get_name(config_path, sub_dir)]
            else:
                framework = SyntheticDataFramework.from_config(config_path, sub_dir)
            all_frameworks.append(framework)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_config = {
                executor.submit(self._load_dataset, framework): framework for framework in all_frameworks
            }
            results = {}
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    data = future.result()
                    results[data.name] = data
                except Exception as exc:
                    print(f"Dataset {config.name} generated an exception: {exc}")
        return results

    def _load_dataset(self, framework: SyntheticDataFramework) -> SyntheticDataFramework:
        framework.generate_dataset()
        return framework

    def run(self) -> None:
        all_frameworks = self.process_datasets_parallel(self.all_sub_frameworks_lazy)
        shuffled_datasets = [
            framework.generated_dataset
            for framework in all_frameworks.values()
            if framework.generated_dataset is not None
        ]
        self.generated_dataset = self.mix(shuffled_datasets)
