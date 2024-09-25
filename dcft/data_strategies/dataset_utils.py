import yaml
from collections import deque
from typing import List, Dict, Optional, Tuple
import os
from pydantic import ValidationError


import ray
from lm_eval.utils import eval_logger
from datasets import Dataset, concatenate_datasets


from dcft.data_strategies.yaml_utils import _get_empty_def, _get_name, _get_len_subcomponents
from engine.operators.configs import OperatorConfig, OperatorSpecificConfig, get_config_class
from engine.operators.operator import create_operator, Operator, ManyShardRefs


class SyntheticDataFramework:
    """
    A framework for creating and managing synthetic data generation processes.

    This class provides methods to parse a DAG (Directed Acyclic Graph) configuration,
    create operators based on the configuration, and execute the data generation process.
    """

    @staticmethod
    def from_config(config_path: str, sub_dir: Optional[Tuple[str, ...]] = None) -> "SyntheticDataFramework":
        """
        Create a SyntheticDataFramework instance from a configuration file.

        Args:
            config_path (str): Path to the configuration file.
            sub_dir (Optional[Tuple[str, ...]]): Subdirectory within the config to use.

        Returns:
            SyntheticDataFramework: An instance of the framework.
        """

        framework = SyntheticDataFramework()
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

        config_type = config.get("type")
        config_class = get_config_class(config_type)
        if config_class is None:
            raise ValueError(f"Unknown config type: {config_type}")
        return config_class(**config)

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
        filtered_pairs = concatenate_datasets([ray.get(shard) for shard in waitables])
        eval_logger.info("Execution completed. Results.")
        self.generated_dataset = filtered_pairs
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

    @staticmethod
    def from_config(config_path: str) -> "DatasetHandler":
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
                framework = SyntheticDataFramework.from_config(config_path, sub_dir)
            all_frameworks.append(framework)

        ray.init(num_cpus=os.cpu_count())
        results = {}
        for framework in all_frameworks:
            results[framework.name] = framework.get_waitables()
        all_waitables = [waitable for waitables in results.values() for waitable in waitables]
        ray.wait(all_waitables, num_returns=len(all_waitables))
        for framework in all_frameworks:
            results[framework.name] = concatenate_datasets([ray.get(shard) for shard in results[framework.name]])
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
