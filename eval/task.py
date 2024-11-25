from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any, Optional, Type
import os
import importlib.util
import sys
import inspect
import logging
from itertools import islice

import torch.distributed as dist
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance


class BaseBenchmark(ABC):
    """Abstract base class for implementing LLM evaluation benchmarks."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def compute(self, model: LM, inputs: List[Instance], do_slice: bool = True) -> List[str]:
        if model.world_size > 1 and do_slice:
            chunk_size = len(inputs) // model.world_size
            start = model.accelerator.process_index * chunk_size
            end = start + chunk_size if model.accelerator.process_index < model.world_size - 1 else len(inputs)
            prompts = inputs[start:end]
        else:
            prompts = inputs

        results = model.generate_until(prompts)
        if model.world_size > 1:
            all_results = [None for _ in range(model.world_size)]
            dist.all_gather_object(all_results, results)

            # Simply concatenate results in rank order
            merged = []
            for sub_results in all_results:
                if sub_results is not None:
                    merged.extend(sub_results)
            return merged
        else:
            return results

    @abstractmethod
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate responses from the model for the benchmark tasks."""
        pass

    @abstractmethod
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the model's responses according to the benchmark's metrics."""
        pass

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """Run the complete benchmark evaluation pipeline."""
        print(f"Running {self.__class__.__name__} benchmark")
        generation_results = self.generate_responses(model)
        evaluation_results = self.evaluate_responses(generation_results)
        return evaluation_results


class TaskManager:
    """
    Enhanced task manager that dynamically loads and manages benchmarks.
    Provides a unified interface for both class-based benchmarks and legacy tasks.
    """

    def __init__(self, benchmarks_dir: str = "chat_benchmarks", **benchmark_kwargs):
        self.logger = logging.getLogger("TaskManager")
        self.tasks: Dict[str, Any] = {}
        self.benchmark_instances: Dict[str, BaseBenchmark] = {}
        self.benchmark_kwargs = benchmark_kwargs

        # Load benchmarks from directory
        self._load_benchmarks(benchmarks_dir)

    def _load_benchmarks(self, benchmarks_dir: str):
        """Dynamically load benchmarks from the specified directory."""
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), benchmarks_dir)

        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if not os.path.isdir(item_path) or item.startswith("__"):
                continue

            eval_path = os.path.join(item_path, "eval_instruct.py")
            if not os.path.exists(eval_path):
                self.logger.warning(f"eval_instruct.py not found in {item}")
                continue

            try:
                # Import the module
                sys.path.insert(0, item_path)
                spec = importlib.util.spec_from_file_location(f"eval.{benchmarks_dir}.{item}.eval_instruct", eval_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.path.pop(0)

                # Find benchmark class
                benchmark_classes = [
                    cls
                    for _, cls in inspect.getmembers(module, inspect.isclass)
                    if (
                        issubclass(cls, BaseBenchmark)
                        and cls != BaseBenchmark
                        and cls.__module__.replace(".", "/") in eval_path
                    )
                ]

                if not benchmark_classes:
                    self.logger.warning(f"No BaseBenchmark subclass found in {item}")
                    continue

                if len(benchmark_classes) > 1:
                    self.logger.warning(f"Multiple benchmark classes found in {item}, using first one")

                benchmark_class = benchmark_classes[0]
                self._register_benchmark(item, benchmark_class)

            except Exception as e:
                self.logger.error(f"Error loading benchmark from {item}: {str(e)}")
                continue

    def _register_benchmark(self, name: str, benchmark_class: Type[BaseBenchmark]):
        """Register a benchmark class and create its instance."""
        try:
            init_params = inspect.signature(benchmark_class.__init__).parameters
            valid_kwargs = {}

            # Only pass kwargs that the benchmark's __init__ accepts
            for param_name, param in init_params.items():
                if param_name in self.benchmark_kwargs:
                    valid_kwargs[param_name] = self.benchmark_kwargs[param_name]
                    self.logger.debug(f"Passing {param_name} to {name} benchmark")

            instance = benchmark_class(**valid_kwargs)

            self.tasks[name] = benchmark_class
            self.benchmark_instances[name] = instance

            self.logger.debug(f"Successfully registered benchmark: {name}")

        except Exception as e:
            self.logger.error(f"Error registering benchmark {name}: {str(e)}")

    def get_list_generate_responses(self, task_list: List[str]) -> List[Callable]:
        """Get list of generate_responses methods for given tasks."""
        methods = []
        for task in task_list:
            if task in self.benchmark_instances:
                methods.append(self.benchmark_instances[task].generate_responses)
            else:
                self.logger.warning(f"Task not found: {task}")
        return methods

    def get_list_evaluates(self, task_list: List[str]) -> List[Callable]:
        """Get list of evaluate_responses methods for given tasks."""
        methods = []
        for task in task_list:
            if task in self.benchmark_instances:
                methods.append(self.benchmark_instances[task].evaluate_responses)
            else:
                self.logger.warning(f"Task not found: {task}")
        return methods

    @property
    def available_tasks(self) -> List[str]:
        """Get list of all available tasks."""
        return list(self.tasks.keys())

    def get_benchmark(self, name: str) -> Optional[BaseBenchmark]:
        """Get a benchmark instance by name."""
        return self.benchmark_instances.get(name)

    def is_valid_task(self, task_name: str) -> bool:
        """Check if a task name is valid."""
        return task_name in self.tasks


def evaluate(
    lm: LM, task_manager: TaskManager, task_list: List[str], verbosity: str = "INFO", **eval_kwargs
) -> Dict[str, Dict]:
    """
    Evaluate the language model on the given tasks.

    Args:
        lm: The language model to evaluate
        task_manager: Task manager containing the benchmarks
        task_list: List of task names to evaluate
        verbosity: Logging verbosity level
        **eval_kwargs: Additional kwargs for evaluation

    Returns:
        Dictionary containing evaluation results for each task
    """
    logger = logging.getLogger("evaluate")
    logger.setLevel(getattr(logging, verbosity))

    results = {"results": {}}

    # Validate tasks
    valid_tasks = [t for t in task_list if task_manager.is_valid_task(t)]
    if len(valid_tasks) != len(task_list):
        invalid_tasks = set(task_list) - set(valid_tasks)
        logger.warning(f"Skipping invalid tasks: {invalid_tasks}")

    if not valid_tasks:
        logger.error("No valid tasks to evaluate")
        return results

    # Run evaluations
    for task_name in valid_tasks:
        try:
            benchmark = task_manager.get_benchmark(task_name)
            if benchmark:
                logger.info(f"Evaluating {task_name}")
                results["results"][task_name] = benchmark.run_benchmark(lm)
        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {str(e)}")
            results["results"][task_name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Initialize task manager
    task_manager = TaskManager()

    # Print available tasks
    print("Available tasks:", task_manager.available_tasks)
