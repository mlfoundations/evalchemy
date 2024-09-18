from typing import List, Dict, Any, Callable
from datasets import load_dataset 

class InstructionGenerator:
    """
    Class for generating instructions.
    """

    def __init__(self, generate_func: Callable[[], List[str]]):
        """
        Initialize the instruction generator with a generate function.

        Args:
            generate_func (Callable[[], List[str]]): Function for instruction generation.
        """
        if is_instance(generate_func, Callable):
            self.generate = generate_func
        elif is_isntance(generate_func, str):
            dataset = load_dataset(generate_func)
            instructions = 



class InstructionSeeder:
    """
    Class for seeding instructions.
    """

    def __init__(self, generate_func: Callable[[], List[str]]):
        """
        Initialize the instruction seeder with a generate function.

        Args:
            generate_func (Callable[[], List[str]]): Function for instruction seeding.
        """
        if is_instance(generate_func, Callable):
            self.generate = generate_func
        elif is_instance(generate_func, str):
            dataset = load_dataset(generate_func)
            if is_instance(dataset, datasets.DatasetDict) and 'train' in dataset:
                instructions = dataset['train']['instruction']

class AnnotationSeeder:
    """
    Class for seeding annotations.
    """

    def __init__(self, generate_func: Callable[[], List[str]]):
        """
        Initialize the annotation seeder with a generate function.

        Args:
            generate_func (Callable[[], List[str]]): Function for seeding annotations.
        """
        self.generate = generate_func

class InstructionFilter:
    """
    Class for filtering instructions.
    """

    def __init__(self, filter_func: Callable[[List[str]], List[str]]):
        """
        Initialize the instruction filter with a filter function.

        Args:
            filter_func (Callable[[List[str]], List[str]]): Function for instruction filtering.
        """
        self.filter = filter_func

class AnnotationGenerator:
    """
    Class for generating annotations.
    """

    def __init__(self, generate_func: Callable[[List[str]], List[Dict[str, Any]]]):
        """
        Initialize the annotation generator with a generate function.

        Args:
            generate_func (Callable[[List[str]], List[Dict[str, Any]]]): Function for annotation generation.
        """
        self.generate = generate_func

class ModelPairFilter:
    """
    Class for filtering model pairs.
    """

    def __init__(self, filter_func: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        """
        Initialize the model pair filter with a filter function.

        Args:
            filter_func (Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]): Function for model pair filtering.
        """
        self.filter = filter_func