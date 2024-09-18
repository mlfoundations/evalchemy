
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class HuggingFaceUploader(ABC):
    """
    Abstract base class for uploading datasets to HuggingFace.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace uploader with a configuration.

        Args:
            config (Dict[str, Any]): Configuration for HuggingFace uploading.
        """
        self.config = config
    
    @abstractmethod
    def upload(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Upload a dataset to HuggingFace.

        Args:
            dataset (List[Dict[str, Any]]): Dataset to upload.
        """
        pass

