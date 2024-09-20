from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DatasetHandler(ABC):
    """
    Abstract base class for handling datasets, including mixing, shuffling, caching, and saving.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset handler with a configuration.

        Args:
            config (Dict[str, Any]): Configuration for dataset handling.
        """
        self.config = config

    @staticmethod
    @abstractmethod
    def mix(datasets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Mix multiple datasets into a single dataset.

        Args:
            datasets (List[List[Dict[str, Any]]]): List of datasets to mix.

        Returns:
            List[Dict[str, Any]]: Mixed dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def shuffle(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Shuffle a dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset to shuffle.

        Returns:
            List[Dict[str, Any]]: Shuffled dataset.
        """
        pass

    @abstractmethod
    def cache(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Cache a dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset to cache.
        """
        pass

    @abstractmethod
    def save(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Save a dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset to save.
        """
        pass
