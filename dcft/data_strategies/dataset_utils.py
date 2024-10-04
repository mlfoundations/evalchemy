import yaml
from collections import deque
from typing import List, Dict, Optional, Tuple
import os
from pydantic import ValidationError
import fsspec

import ray
from lm_eval.utils import eval_logger
from datasets import Dataset, concatenate_datasets


from dcft.data_strategies.yaml_utils import _get_empty_def, _get_name, _get_len_subcomponents
from engine.operators.configs import OperatorConfig, OperatorSpecificConfig, get_config_class, hash_operator_config_list
from engine.operators.operator import create_operator, Operator, ManyShardRefs, LoadFromCacheOperator


class SyntheticDataFramework:
    """
    A framework for creating and managing synthetic data generation processes.

    This class provides methods to parse a DAG (Directed Acyclic Graph) configuration,
    create operators based on the configuration, and execute the data generation process.
    """

    def __init__(self) -> None:
        self.overwrite_cache = False

    @staticmethod
    def from_config(
        config_path: str,
        sub_dir: Optional[Tuple[str, ...]] = None,
        cache_dir: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        overwrite_cache: Optional[str] = None,
    ) -> "SyntheticDataFramework":
        """
        Create a SyntheticDataFramework instance from a configuration file.

        Args:
            config_path (str): Path to the configuration file.
            sub_dir (Optional[Tuple[str, ...]]): Subdirectory within the config to use.

        Returns:
            SyntheticDataFramework: An instance of the framework.
        """
        framework = SyntheticDataFramework()
        framework.cache_dir = cache_dir
        framework.fs = fs
        framework.overwrite_cache = overwrite_cache
        framework.parse_dag(config_path, sub_dir)
        return framework

    def parse_dag(self, config_path: str, sub_dir: Optional[Tuple[str, ...]] = None) -> List[Operator]:
        """
        Takes the configuration file and loads the individual operators into a single data generation pipeline using a Directed Acyclic Graph Structure

        Args:
            config_path (str): Path to the configuration file.
            sub_dir (Optional[Tuple[str, ...]]): Subdirectory within the config to use.

        Returns:
            List[Operator]: A list of created operators in topological graph order.

        Raises:
            ValueError: If there are duplicate operator IDs or invalid configurations.
        """

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
                specific_config = self.parse_specific_config(op["config"])
                operator_config = OperatorConfig(id=op_id, input_ids=op.get("input_ids", []), config=specific_config)
                operator = create_operator(operator_config)
                operators.append(operator)

            except ValidationError as e:
                breakpoint()
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
            raise ValueError("The graph contains a cycle or input_ids not matching")

        # Create a mapping of operator IDs to Operator objects
        op_map = {op.id: op for op in operators}

        # Return the sorted list of Operator objects
        self.linearized_dag_functions = [op_map[op_id] for op_id in sorted_operators]
        ancestor_configs = self.get_ancestor_configs(op_map, sorted_operators)
        self.map_op_id_to_dag_hash = {
            op: hash_operator_config_list(ancestor_configs[op.id]) for op in self.linearized_dag_functions
        }

        for idx in range(len(self.linearized_dag_functions)):
            op = self.linearized_dag_functions[idx]
            if self.fs.exists(f"{self.cache_dir}/{self.map_op_id_to_dag_hash[op]}"):
                print(f"Found in cache for op: {op.id}")
                self.linearized_dag_functions[idx] = LoadFromCacheOperator(
                    op.id, op.input_ids, f"{self.cache_dir}/{self.map_op_id_to_dag_hash[op]}"
                )

    def get_ancestor_operators(
        self, op_map: Dict[str, Operator], sorted_operators: List[Operator]
    ) -> Dict[str, List[Operator]]:
        # Initialize a dictionary to store the ancestor operators for each op
        ancestor_operators = {}

        def get_op_ancestors(op_id):
            # If we've already computed this op's ancestors, return them
            if op_id in ancestor_operators:
                return ancestor_operators[op_id]

            op = op_map[op_id]

            # If this op has no parents, its ancestor list contains only itself
            if not op.input_ids:
                ancestor_operators[op_id] = [op]
                return ancestor_operators[op_id]

            # Get ancestors for all parents, preserving order
            parents_ancestors = []
            for parent_id in op.input_ids:
                parents_ancestors.extend(get_op_ancestors(parent_id))

            # Remove duplicates while preserving order
            unique_ancestors = []
            seen = set()
            for ancestor in parents_ancestors:
                if ancestor.id not in seen:
                    unique_ancestors.append(ancestor)
                    seen.add(ancestor.id)

            # Add this op to the end of the unique ancestors
            ancestor_operators[op_id] = unique_ancestors + [op]
            return ancestor_operators[op_id]

        # Iterate through the sorted operators and compute their ancestors
        for op_id in sorted_operators:
            get_op_ancestors(op_id)

        return ancestor_operators

    def get_ancestor_configs(
        self, op_map: Dict[str, Operator], sorted_operators: List[Operator]
    ) -> Dict[str, OperatorSpecificConfig]:
        ancestor_operators = self.get_ancestor_operators(op_map, sorted_operators)
        ancestor_configs = {op_id: [op.config for op in ops] for op_id, ops in ancestor_operators.items()}
        return ancestor_configs

    def parse_specific_config(self, config: dict) -> OperatorSpecificConfig:
        """
        Parse the specific configuration for an operator.

        Args:
            config (dict): The configuration dictionary for the operator.

        Returns:
            OperatorSpecificConfig: The parsed configuration object.

        Raises:
            ValueError: If the config type is unknown.
        """
        try:
            config_type = config.get("type")
            config_class = get_config_class(config_type)
        
            if config_class is None:
                raise ValueError(f"Unknown config type: {config_type}")
            return config_class(**config)
        except:
            breakpoint()

    def get_waitables(self) -> ManyShardRefs:
        """
        Execute the operators in the DAG and return a promise of the list of shards at the end of the data generation process.

        Returns:
            ManyShardRefs: References to the output shards of the data generation process.
        """

        datas = {}
        for idx, operator in enumerate(self.linearized_dag_functions):
            input_datas = {input_id: datas[input_id] for input_id in operator.input_ids}
            curr_op_output = operator.execute(input_datas)
            datas[operator.id] = curr_op_output
            if idx == len(self.linearized_dag_functions) - 1:
                waitables = curr_op_output
        return waitables

    def run(self) -> None:
        """
        Run the entire data generation process.

        This method initializes Ray, executes the DAG, and processes the results.
        """

        ray.init(num_cpus=os.cpu_count())
        waitables = self.get_waitables()
        ray.wait(waitables, num_returns=len(waitables))
        try:
            filtered_pairs = concatenate_datasets([ray.get(shard) for shard in waitables])
        except:
            breakpoint()
        eval_logger.info("Execution completed. Results.")
        self.generated_dataset = filtered_pairs

        for op in self.linearized_dag_functions:
            if not isinstance(op, LoadFromCacheOperator):
                op.cleanup(
                    self.fs,
                    cache_dir=f"{self.cache_dir}/{self.map_op_id_to_dag_hash[op]}",
                    overwrite_cache=self.overwrite_cache,
                )
        ray.shutdown()


class DatasetHandler:
    """
    A class for handling multiple datasets and performing operations on them.

    This class can load multiple SyntheticDataFramework instances, process them in parallel,
    and mix the resulting datasets.
    """

    def __init__(self, sub_frameworks_lazy: List[Tuple[str, Tuple[str, int]]]):
        """
        Initialize the DatasetHandler.

        Args:
            sub_frameworks_lazy (List[Tuple[str, Tuple[str, int]]]): A list of tuples containing
                the config path and subdirectory for each sub-framework which will be loaded lazily.
        """

        self.all_sub_frameworks_lazy = sub_frameworks_lazy
        self.max_workers = os.cpu_count()
        self.shuffle_seed = 42
        self.all_loaded_frameworks: Dict[str, SyntheticDataFramework] = {}

    def set_cache_dir(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    def set_overwrite_cache(self, overwrite_cache: bool) -> None:
        self.overwrite_cache = overwrite_cache

    def set_fs(self, fs: fsspec.AbstractFileSystem):
        self.fs = fs

    @staticmethod
    def from_config(
        config_path: str,
        cache_dir: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        overwrite_cache: Optional[str] = None,
    ) -> "DatasetHandler":
        """
        Create a DatasetHandler instance from a configuration file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            DatasetHandler: An instance of the DatasetHandler.
        """
        num_components = _get_len_subcomponents(config_path)
        all_sub_frameworks_lazy = []
        for index in range(num_components):
            all_sub_frameworks_lazy.append((config_path, ("dataset_mix", index)))
        dataset_handler = DatasetHandler(all_sub_frameworks_lazy)
        dataset_handler.cache_dir = cache_dir
        dataset_handler.fs = fs
        dataset_handler.overwrite_cache = overwrite_cache
        dataset_handler.name = _get_name(config_path)
        return dataset_handler

    def mix(self, datasets: List[Dataset]) -> Dataset:
        """
        Mix multiple datasets into a single, shuffled dataset.

        Args:
            datasets (List[Dataset]): A list of datasets to mix.

        Returns:
            Dataset: The mixed and shuffled dataset.
        """
        print("Starting dataset mixing process...")

        combined_dataset = concatenate_datasets(datasets)
        print(f"Combined {len(datasets)} datasets, total items: {len(combined_dataset)}")

        shuffled_dataset = combined_dataset.shuffle(seed=self.shuffle_seed)
        print("Dataset shuffled")

        return shuffled_dataset

    def process_datasets_parallel(self, dataset_configs: List[Tuple[str, Tuple[str, int]]]) -> Dict[str, Dataset]:
        """
        Process multiple datasets in parallel using Ray. If an individual dataset is defined in another yaml file, it will fetch that definition.

        Args:
            dataset_configs (List[Tuple[str, Tuple[str, int]]]): A list of Synthetic Dataset Framework configurations.

        Returns:
            Dict[str, Dataset]: A dictionary mapping dataset names to processed datasets.

        Raises:
            ValueError: If a required framework is not defined or loaded.
        """
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
                framework = SyntheticDataFramework.from_config(
                    config_path,
                    sub_dir=sub_dir,
                    cache_dir=self.cache_dir,
                    fs=self.fs,
                    overwrite_cache=self.overwrite_cache,
                )
            all_frameworks.append(framework)

        ray.init(num_cpus=os.cpu_count())
        results = {}
        for framework in all_frameworks:
            results[framework.name] = framework.get_waitables()
        all_waitables = [waitable for waitables in results.values() for waitable in waitables]
        ray.wait(all_waitables, num_returns=len(all_waitables))
        for framework in all_frameworks:
            results[framework.name] = concatenate_datasets([ray.get(shard) for shard in results[framework.name]])

        for framework in all_frameworks:
            for op in framework.linearized_dag_functions:
                if not isinstance(op, LoadFromCacheOperator):
                    op.cleanup(
                        self.fs,
                        cache_dir=f"{self.cache_dir}/{framework.map_op_id_to_dag_hash[op]}",
                        overwrite_cache=self.overwrite_cache,
                    )

        ray.shutdown()
        return results

    def run(self) -> None:
        """
        Run the entire dataset processing pipeline.

        This method processes all datasets in parallel, mixes them, and stores the result.
        """
        all_datasets = self.process_datasets_parallel(self.all_sub_frameworks_lazy)
        for name in all_datasets.keys():
            all_datasets[name] = all_datasets[name].add_column("_subdataset_name", [name] * len(all_datasets[name]))
        shuffled_datasets = [dataset for dataset in all_datasets.values()]
        self.generated_dataset = self.mix(shuffled_datasets)
