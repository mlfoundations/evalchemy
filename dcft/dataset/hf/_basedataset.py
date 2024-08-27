import json
from datasets import load_dataset

class BaseFTDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        try:
            with open(data_path, 'r') as f:
                self.data = json.loads(f)
        except:
            self.data = load_dataset(data_path)

        self.system_prompts = []
        self.user_prompts = []
        self.annotations_gtruth = []
        self.annotations = []

    def __len__(self):
        return len(self.data)

    def __idx__(self, idx):
        if "train" in self.data:
            return self.data["train"][idx]
        else:
            return self.data[idx]