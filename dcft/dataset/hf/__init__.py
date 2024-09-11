from dcft.dataset.hf.codealpaca import CodeAlpacaFTDataset
from dcft.dataset.hf.codeglaive import CodeGlaiveFTDataset
dataset_map = {
    "sahil2801/CodeAlpaca-20k": CodeAlpacaFTDataset,
    "glaiveai/glaive-code-assistant": CodeGlaiveFTDataset,
}

def get_dataclass_from_path(data_path):
    return dataset_map[data_path](data_path)