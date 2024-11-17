import json
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset


class BaseFTDataset:
    def __init__(self, data_path: str) -> None:
        self.data_path: str = data_path
        try:
            with open(data_path, "r") as f:
                self.data: Union[Dict[str, Any], List[Any]] = json.loads(f)
        except:
            self.data = load_dataset(data_path)

        self.system_prompts: List[str] = []
        self.user_prompts: List[str] = []
        self.annotations_original: List[str] = []
        self.annotations: List[Any] = []
        self.batch_objects: Optional[Any] = None

    def __len__(self) -> int:
        return len(self.data)

    def __idx__(self, idx: int) -> Union[Dict[str, Any], Any]:
        if "train" in self.data:
            return self.data["train"][idx]
        else:
            return self.data[idx]
