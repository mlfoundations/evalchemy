import os
from typing import Callable, Dict, List, Tuple, Any
from pathlib import Path
from engine.operators.operator import (
    parse_yaml_config,
)


def remove_prefix(path: str) -> str:
    # Normalize paths to handle different path separators
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_current = os.path.normpath(current_dir)
    norm_path = os.path.normpath(path)

    # Check if the path starts with the current directory
    if norm_path.startswith(norm_current):
        # Remove the prefix and the leading separator
        relative_path = norm_path[len(norm_current) :].lstrip(os.sep)
        return relative_path
    return path


def yaml_handler(path: str) -> Tuple[str, str]:
    """Process single YAML file and return name-path pair"""
    return Path(path).stem, path


def walk_directory(
    directory: str, file_extensions: tuple = (".yaml", ".yml"), skip_dirs: tuple = ("__pycache__",)
) -> Dict[str, Any]:
    """
    Recursively walk through directory and process files with specified extensions.

    Args:
        directory (str): Root directory path to walk through
        file_extensions (tuple): File extensions to process
        skip_dirs (tuple): Directory names to skip

    Returns:
        Dict[str, Any]: Dictionary of processed results

    Example:
        def yaml_handler(path: str) -> Tuple[str, str]:
            config = parse_yaml_config(path)
            return config["name"], path

        results = walk_directory("/path/to/dir", yaml_handler)
    """
    results = {}

    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith(file_extensions):
                key, value = yaml_handler(entry.path)
                if key in results:
                    raise ValueError(
                        f"Duplicate key '{key}' found in {entry.path}. " f"Original definition in {results[key]}"
                    )
                results[key] = value

            elif entry.is_dir() and entry.name not in skip_dirs:
                try:
                    subfolder_results = walk_directory(entry.path, file_extensions, skip_dirs)
                    # Check for duplicates before updating
                    for key, value in subfolder_results.items():
                        if key in results:
                            raise ValueError(
                                f"Duplicate key '{key}' found in {value}. " f"Original definition in {results[key]}"
                            )
                    results.update(subfolder_results)
                except PermissionError:
                    print(f"Warning: Permission denied accessing directory {entry.path}")
                except Exception as e:
                    print(f"Warning: Error processing directory {entry.path}: {str(e)}")

    except PermissionError:
        print(f"Warning: Permission denied accessing directory {directory}")
    except Exception as e:
        print(f"Warning: Error accessing directory {directory}: {str(e)}")

    return results
