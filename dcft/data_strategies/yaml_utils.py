import yaml
from typing import List, Any, Optional


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    """
    A custom YAML loader that ignores unknown tags.
    """

    def ignore_unknown(self, node: Any) -> None:
        return None


SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)


def _get_empty_def(file_path: str, subdir: List[str]) -> bool:
    """
    Check if a specific subdirectory in a YAML file contains an empty definition.

    Args:
        file_path (str): Path to the YAML file.
        subdir (List[str]): List of keys representing the subdirectory path.

    Returns:
        bool: True if the subdirectory contains only one key, False otherwise.
    """
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
    """
    Check if a YAML file contains a 'dataset_mix' key at the top level.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        bool: True if 'dataset_mix' is present, False otherwise.

    Raises:
        yaml.YAMLError: If there's an error parsing the YAML file.
        IOError: If there's an error opening the file.
    """
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
    """
    Get the number of datasets in the 'dataset_mix' section of a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        int: The number of datasets in the 'dataset_mix' section.
    """
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    return len(config["dataset_mix"])


def _get_name(file_path: str, sub_dir: Optional[List[str]] = None) -> str:
    """
    Get the 'name' field from a YAML file, optionally from a specific subdirectory.

    Args:
        file_path (str): Path to the YAML file.
        sub_dir (Optional[List[str]]): List of keys representing the subdirectory path.

    Returns:
        str: The value of the 'name' field.
    """
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    if sub_dir is not None:
        for key in sub_dir:
            config = config[key]

    return config["name"]
