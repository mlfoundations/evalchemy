from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DatasetMixer(ABC):
    """
    Abstract base class for mixing datasets.
    """

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


class DatasetShuffler(ABC):
    """
    Abstract base class for shuffling datasets.
    """

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


class DatasetCache(ABC):
    """
    Abstract base class for caching datasets.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset cache with a configuration.

        Args:
            config (Dict[str, Any]): Configuration for dataset caching.
        """
        self.config = config

    @abstractmethod
    def cache(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Cache a dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset to cache.
        """
        pass


class DatasetSaver(ABC):
    """
    Abstract base class for saving datasets.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset saver with a configuration.

        Args:
            config (Dict[str, Any]): Configuration for dataset saving.
        """
        self.config = config

    @abstractmethod
    def save(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Save a dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset to save.
        """
        pass
