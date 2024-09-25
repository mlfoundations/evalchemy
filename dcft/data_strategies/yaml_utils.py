import yaml
from typing import List, Any, Optional

class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node: Any) -> None:
        return None
    
SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

def _get_empty_def(file_path: str, subdir: List[str]) -> bool:
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    for key in subdir:
        config = config[key]

    if len(config.keys()) == 1:
        return True
    else:
        return False

def check_dataset_mix_in_yaml(file_path: str) -> bool:
    try:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
        return "dataset_mix" in config if isinstance(config, dict) else False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return False
    except IOError as e:
        print(f"Error opening file {file_path}: {e}")
        return False


def _get_len_subcomponents(file_path: str) -> int:
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    return len(config["dataset_mix"])


def _get_name(file_path: str, sub_dir: Optional[List[str]] = None) -> str:
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    if sub_dir is not None:
        for key in sub_dir:
            config = config[key]

    return config["name"]
