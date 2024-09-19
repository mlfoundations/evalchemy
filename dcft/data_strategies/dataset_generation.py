from typing import List, Dict, Any, Callable
from datasets import load_dataset 
import re 

class InstructionGenerator:
    """
    Class for generating instructions.
    """

    def __init__(self, generate_info: Callable[[], List[str]]):
        """
        Initialize the instruction generator with a generate function.

        Args:
            generate_info (Callable[[], List[str]]): Function for instruction generation.
        """
        if isinstance(generate_info, Callable):
            self.generate = generate_info
        elif isinstance(generate_info, str):
            dataset_name, split_name, instruction_name = generate_info.split("\\")
            dataset = load_dataset(dataset_name)
            
            instructions = list(dataset[split_name].map(lambda x: [instruction_name]))
            self.generate = lambda: instructions



class InstructionSeeder:
    """
    Class for seeding instructions.
    """

    def __init__(self, generate_info: Callable[[], List[str]]):
        """
        Initialize the instruction seeder with a generate function.

        Args:
            generate_info (Callable[[], List[str]]): Function for instruction seeding.
        """
        if generate_info is None:
            self.generate = lambda: []
        elif isinstance(generate_info, Callable):
            self.generate = generate_info
        elif isinstance(generate_info, str):
            split_string = generate_info.split("//")
            dataset_name, split_name, seed_name = split_string[0], split_string[1], split_string[2]
            dataset = load_dataset(dataset_name)
            seeds = dataset[split_name][seed_name]
            self.generate = lambda: seeds

        

class AnnotationSeeder:
    """
    Class for seeding annotations.
    """

    def __init__(self, generate_info: Callable[[], List[str]]):
        """
        Initialize the annotation seeder with a generate function.

        Args:
            generate_info (Callable[[], List[str]]): Function for seeding annotations.
        """
        if generate_info is None:
            self.generate = lambda x: x
        else:
            self.generate = generate_info

class InstructionFilter:
    """
    Class for filtering instructions.
    """

    def __init__(self, filter_info: Callable[[List[str]], List[str]]):
        """
        Initialize the instruction filter with a filter function.

        Args:
            filter_info (Callable[[List[str]], List[str]]): Function for instruction filtering.
        """
        if filter_info is None:
            self.filter = lambda x: x
        else:
            self.filter = filter_info

class AnnotationGenerator:
    """
    Class for generating annotations.
    """

    def __init__(self, generate_info: Callable[[List[str]], List[Dict[str, Any]]]):
        """
        Initialize the annotation generator with a generate function.

        Args:
            generate_info (Callable[[List[str]], List[Dict[str, Any]]]): Function for annotation generation.
        """
        self.generate = generate_info

class ModelPairFilter:
    """
    Class for filtering model pairs.
    """

    def __init__(self, filter_info: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        """
        Initialize the model pair filter with a filter function.

        Args:
            filter_info (Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]): Function for model pair filtering.
        """
        if filter_info is None:
            self.filter = lambda x: x
        else:
            self.filter = filter_info