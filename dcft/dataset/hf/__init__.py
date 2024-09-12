from dcft.dataset.hf.codealpaca import CodeAlpacaFTDataset

dataset_map = {"sahil2801/CodeAlpaca-20k": CodeAlpacaFTDataset}


def get_dataclass_from_path(data_path):
    return dataset_map[data_path](data_path)
