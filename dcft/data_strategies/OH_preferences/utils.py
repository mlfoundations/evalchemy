from datasets import Dataset
from typing import Dict


def filter(ex: Dict) -> Dict:
    ex["conversations"] = [{"from": "human", "value": ex["prompt"]}]
    ex["chosen"] = {"from": "gpt", "value": ex["chosen"][1]["content"]}
    ex["rejected"] = {"from": "gpt", "value": ex["rejected"][1]["content"]}
    return ex


def parse_oh_pref(dataset: Dataset) -> Dataset:
    return dataset.map(filter)
