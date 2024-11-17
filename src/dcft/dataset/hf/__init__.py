from dcft.dataset.hf.alpaca import AlpacaFTDataset
from dcft.dataset.hf.codeglaive import CodeGlaiveFTDataset

HF_DATASET_MAP = {
    "sahil2801/CodeAlpaca-20k": AlpacaFTDataset,
    "glaiveai/glaive-code-assistant": CodeGlaiveFTDataset,
    "garage-bAInd/Open-Platypus": AlpacaFTDataset,
    "causal-lm/cot_alpaca_gpt4": AlpacaFTDataset,
    "teknium/GPT4-LLM-Cleaned": AlpacaFTDataset,
    "xzuyn/lima-multiturn-alpaca": AlpacaFTDataset,
}


def get_dataclass_from_path(data_path):
    return HF_DATASET_MAP[data_path](data_path)
