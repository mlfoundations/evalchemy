import ast
import base64
import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Callable, Dict, List, Union

import asttokens
import code2flow
import networkx as nx


class HashCodeHelper:
    def __init__(self):
        self.base_dirs = ["dcft/data_strategies", "dcft/external_repositories"]
        logging.info(f"Initializing data generation hashing...")
        python_files = self.get_python_files(self.base_dirs)
        output_file = self.run_code2flow(python_files)
        self.json_data = self.parse_code2flow_output(output_file)
        self.call_graph = self.build_call_graph(self.json_data)

    def get_python_files(self, directories: List[str]) -> List[str]:
        """
        Recursively find all Python files in the given directories.

        Args:
            directories (List[str]): List of directory paths to search.

        Returns:
            List[str]: List of paths to Python files found.
        """
        python_files = []
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".py"):
                        python_files.append(os.path.join(root, file))
        return python_files

    def run_code2flow(self, python_files: List[str]) -> str:
        """
        Run code2flow on the given Python files and save the output to a temporary file.

        Args:
            python_files (List[str]): List of Python file paths to analyze.

        Returns:
            str: Path to the temporary file containing the code2flow output.
        """
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
            temp_file_path = temp_file.name
            code2flow.code2flow(python_files, temp_file_path, level=logging.ERROR)
        return temp_file_path

    def parse_code2flow_output(self, temp_file_path: str) -> Dict:
        """
        Parse the JSON output from code2flow.

        Args:
            temp_file_path (str): Path to the temporary file containing code2flow output.

        Returns:
            Dict: Parsed JSON data from code2flow.
        """
        with open(temp_file_path, "r") as f:
            json_data = json.load(f)
        os.unlink(temp_file_path)
        return json_data

    def build_call_graph(self, json_data: Dict) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from the code2flow JSON data.

        Args:
            json_data (Dict): Parsed JSON data from code2flow.

        Returns:
            nx.DiGraph: NetworkX directed graph representing the call graph.
        """
        G = nx.DiGraph()
        for node_id, node_data in json_data["graph"]["nodes"].items():
            G.add_node(node_id, **node_data)
        for edge in json_data["graph"]["edges"]:
            G.add_edge(edge["source"], edge["target"])
        return G

    @staticmethod
    def get_function_source(file_path: str, function_name: str) -> str:
        """
        Extract the source code of a specific function from a file.

        Args:
            file_path (str): Path to the Python file.
            function_name (str): Name of the function to extract.

        Returns:
            str: Source code of the function, or an empty string if not found.
        """
        with open(file_path, "r") as f:
            content = f.read()
        atok = asttokens.ASTTokens(content, parse=True)
        for node in ast.walk(atok.tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return atok.get_text(node)
        return ""

    @staticmethod
    def split_path_and_function(full_path: str) -> tuple[str, str]:
        """
        Split a full path into relative path and function name.

        Args:
            full_path (str): The full path including the function name.

        Returns:
            tuple[str, str]: A tuple containing (relative_path, function_name).
        """
        parts = full_path.split(".")
        function_name = parts[-1]
        relative_path = "/".join(parts[:-1])
        return relative_path, function_name

    def gather_function_code(self, G: nx.DiGraph, start_function: str) -> str:
        """
        Gather the source code of all functions reachable from the start function in the call graph.

        Args:
            G (nx.DiGraph): NetworkX directed graph representing the call graph.
            start_function (str): Starting function node in the call graph.

        Returns:
            str: Concatenated source code of all reachable functions.
        """
        code = ""
        visited = set()
        stack = [start_function]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node_data = G.nodes[node_id]
            relative_path, function_name = node_data["name"].split("::")
            code += self.get_function_source(relative_path, function_name) + "\n\n"
            for neighbor in G.successors(node_id):
                if neighbor not in visited:
                    stack.append(neighbor)
        return code

    def find_node_by_name(self, G: nx.DiGraph, name: str) -> Union[str, None]:
        """
        Find a node in the graph by its name attribute.

        Args:
            G (nx.DiGraph): NetworkX directed graph to search.
            name (str): Name to search for.

        Returns:
            Union[str, None]: Node identifier if found, None otherwise.
        """
        for node, data in G.nodes(data=True):
            if data["name"] == name:
                return node
        return None

    @staticmethod
    def format_module_path(module_string: str) -> str:
        """
        Format a module string into a file path with function name.

        Args:
            module_string (str): Module string to format.

        Returns:
            str: Formatted file path with function name.
        """
        module_parts = module_string.rsplit(".", 1)
        if len(module_parts) == 1:
            return f"{module_string}.py"
        file_path = module_parts[0].replace(".", "/")
        return f"dcft/{file_path}.py::{module_parts[1]}"

    def hash_function(self, input_function_name: str) -> str:
        """
        Compute and print the hash of a function and its dependencies.

        Args:
            input_function_name (str): Name of the starting function.

        Returns:
            str: Hashed function code including dependencies
        """
        start_function_name = self.format_module_path(input_function_name)
        start_node = self.find_node_by_name(self.call_graph, start_function_name)
        if start_node is None:
            relative_path, file_name = self.split_path_and_function(input_function_name)
            function_code = self.get_function_source(f"dcft/{relative_path}.py", file_name)
        else:
            function_code = self.gather_function_code(self.call_graph, start_node)
        function_hash = self.hash_value(function_code)
        return function_hash

    def hash_value(self, value: Any, key=None) -> str:
        """
        Create a hash string for a value, handling unhashable types.

        Args:
            value (Any): The value to hash.
            key (str, optional): Type of value. Defaults to None.

        Returns:
            str: A hash string value.
        """
        if isinstance(value, Callable) or key == "function":
            return self.hash_function(value)
        elif isinstance(value, (list, tuple)):
            return f"list:{','.join(self.hash_value(item) for item in value)}"
        elif isinstance(value, dict):
            return f"dict:{','.join(f'{k}:{self.hash_value(v)}' for k, v in sorted(value.items()))}"
        elif isinstance(value, set):
            return f"set:{','.join(sorted(self.hash_value(item) for item in value))}"
        elif hasattr(value, "__dict__"):
            return f"obj:{self.hash_value(value.__dict__)}"
        else:
            return f"val:{hashlib.sha256(str(value).encode()).hexdigest()[:16]}"

    def hash_operator_config_list(self, config_list: List[Any]) -> str:
        """
        Hash a list of OperatorSpecificConfig objects to a sanitized string.

        Args:
            config_list (List[Any]): List of OperatorSpecificConfig objects to hash.

        Returns:
            str: Sanitized string hash value of the list.
        """
        config_strings = [
            ",".join(f"{k}:{self.hash_value(v, key=k)}" for k, v in config.dict().items()) for config in config_list
        ]
        full_string = "|".join(config_strings)
        hash_object = hashlib.sha256(full_string.encode())
        return base64.urlsafe_b64encode(hash_object.digest()).decode().rstrip("=")
