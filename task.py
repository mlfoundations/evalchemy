from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any
import os
import importlib.util
import sys

from lm_eval.api.model import LM

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


def import_eval_instructs() -> Dict[str, "Task"]:
    """
    Dynamically import evaluation instructions from chat_benchmarks directory.

    Returns:
        Dict[str, Task]: A dictionary mapping task names to Task instances.
    """
    current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_benchmarks")
    tasks = {}

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and not item.startswith("__"):
            eval_instruct_path = os.path.join(item_path, "eval_instruct.py")
            if os.path.exists(eval_instruct_path):
                try:
                    sys.path.insert(0, item_path)
                    spec = importlib.util.spec_from_file_location(
                        f"eval.chat_benchmarks.{item}.eval_instruct", eval_instruct_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    sys.path.pop(0)
                    if hasattr(module, "eval_instruct"):
                        # Dynamically create a Task subclass
                        task_class = type(
                            f"{item}Task",
                            (Task,),
                            {
                                "eval_instruct": staticmethod(module.eval_instruct),
                                "evaluate": staticmethod(module.evaluate),
                            },
                        )
                        tasks[item] = task_class()
                    else:
                        print(f"Warning: eval_instruct function not found in {item}")
                except Exception as e:
                    print(f"Warning: Could not import eval_instruct from {item}: {str(e)}")
            else:
                print(f"Warning: eval_instruct.py not found in {item}")
    print(tasks)
    return tasks


class TaskManager:
    """Manages the collection of evaluation tasks."""

    def __init__(self):
        self.tasks: Dict[str, Task] = import_eval_instructs()

    def get_list_eval_instructs(self, task_list: List[str]) -> List[Callable[[LM], Dict[str, Any]]]:
        """
        Get a list of eval_instruct methods for the specified tasks.

        Args:
            task_list (List[str]): List of task names.

        Returns:
            List[Callable[[LM], Dict[str, Any]]]: List of eval_instruct methods.
        """
        return [self.tasks[task].eval_instruct for task in task_list]

    def get_list_evaluates(self, task_list: List[str]) -> List[Callable[[Dict[str, Any]], Dict[str, Any]]]:
        """
        Get a list of evaluate methods for the specified tasks.

        Args:
            task_list (List[str]): List of task names.

        Returns:
            List[Callable[[Dict[str, Any]], Dict[str, Any]]]: List of evaluate methods.
        """
        return [self.tasks[task].evaluate for task in task_list]


class Task(ABC):
    """Abstract base class for evaluation tasks."""

    @abstractmethod
    def eval_instruct(self, model: LM) -> Dict[str, Any]:
        """
        Abstract method for evaluation instruction.

        Args:
            model (LM): The language model to be evaluated.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement eval_instruct method")

    @abstractmethod
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method for evaluation.

        Args:
            results (Dict[str, Any]): The results to be evaluated.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation metrics.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


if __name__ == "__main__":
    task_manager = TaskManager()
