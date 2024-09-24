import yaml
import importlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from datasets import Dataset, concatenate_datasets
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from lm_eval.utils import eval_logger
import pandas as pd
from datasets import load_dataset
import ray
import yaml
from pydantic import ValidationError

from engine.operators import (
    Operator,
    OperatorConfig,
    OperatorSpecificConfig,
    create_operator
)
from engine.operators.sharding import ShardOperatorConfig, ShardDatasetOperator,  MergeOperatorConfig, MergeShardsOperator
from engine.operators.registry import get_config_class
from collections import deque
class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node: Any) -> None:
        return None
    
SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

def check_dataset_mix_in_yaml(file_path: str) -> bool:
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
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    return len(config["dataset_mix"])


def _get_name(file_path: str, sub_dir: Optional[List[str]] = None) -> str:

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
        framework.parse_dag(config_path, sub_dir)
        return framework
        
    def parse_dag(self, config_path: str, sub_dir: Optional[Tuple[str, ...]] = None) -> List[Operator]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if sub_dir is not None:
            for key in sub_dir:
                config = config[key]
        
        self.name = config["name"]
        operators = []

        seen_ids = set()
        for op in config["operators"]:
            op_id = op["id"]
            if op_id in seen_ids:
                raise ValueError(f"Duplicate operator ID found: {op_id}")
            seen_ids.add(op_id)

            try:
                specific_config = self.parse_specific_config(op['config'])
                operator_config = OperatorConfig(id=op_id, input_ids=op.get("input_ids", []), config=specific_config)
                operator = create_operator(operator_config)
                operators.append(operator)

            except ValidationError as e:
                raise ValueError(f"Invalid configuration for operator {op_id}: {str(e)}")

        # Create a graph representation
        graph = {op.id: set(op.input_ids) for op in operators}

        # Topological sort
        sorted_operators = []
        no_incoming = deque([op.id for op in operators if not graph[op.id]])

        while no_incoming:
            node = no_incoming.popleft()
            sorted_operators.append(node)

            for neighbor in list(graph.keys()):
                if node in graph[neighbor]:
                    graph[neighbor].remove(node)
                    if not graph[neighbor]:
                        no_incoming.append(neighbor)

        if len(sorted_operators) != len(operators):
            raise ValueError("The graph contains a cycle")

        # Create a mapping of operator IDs to Operator objects
        op_map = {op.id: op for op in operators}

        # Return the sorted list of Operator objects
        self.linearized_dag_functions = [op_map[op_id] for op_id in sorted_operators]

    def parse_specific_config(self, config: dict) -> OperatorSpecificConfig:
        config_type = config.get("type")
        config_class = get_config_class(config_type)
        if config_class is None:
            raise ValueError(f"Unknown config type: {config_type}")
        try:
            return config_class(**config)
        except:
            breakpoint()
    
    def generate_dataset(self) -> None:
        ray.init()
        datas = {}
        try:
            for operator in self.linearized_dag_functions:
                input_datas = {input_id: datas[input_id] for input_id in operator.input_ids}
                datas[operator.id] = operator.execute(input_datas)

            # Wait for all tasks to complete and retrieve results
            waitables = [data_shard for data in datas.values() for data_shard in data]
            ray.wait(waitables, num_returns=len(waitables))
            filtered_pairs = [ray.get(shard) for shard in waitables]
            eval_logger.info("Execution completed. Results.")
            breakpoint()
        finally:
            # Shut down Ray
            ray.shutdown()
        
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
