from dcft.dataset.hf.codealpaca import CodeAlpacaFTDataset
from dcft.dataset.hf.codeglaive import CodeGlaiveFTDataset
from dcft.dataset.hf.platypus import PlatypusFTDataset
from dcft.dataset.hf.cotalpaca import COTAlpacaFTDataset

dataset_map = {
    "sahil2801/CodeAlpaca-20k": CodeAlpacaFTDataset,
    "glaiveai/glaive-code-assistant": CodeGlaiveFTDataset,
    "garage-bAInd/Open-Platypus": PlatypusFTDataset,
    "causal-lm/cot_alpaca_gpt4": COTAlpacaFTDataset,  
    "teknium/GPT4-LLM-Cleaned": COTAlpacaFTDataset,
}


def get_dataclass_from_path(data_path):
    return dataset_map[data_path](data_path)
