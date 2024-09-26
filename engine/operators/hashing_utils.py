import os
import code2flow
import json
import hashlib
import networkx as nx
import tempfile
import ast
import asttokens


def get_python_files(directories):
    python_files = []
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
    return python_files


def run_code2flow(python_files):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
        temp_file_path = temp_file.name
        code2flow.code2flow(python_files, temp_file_path)
    return temp_file_path


def parse_code2flow_output(temp_file_path):
    with open(temp_file_path, "r") as f:
        json_data = json.load(f)

    # Clean up the temporary file
    os.unlink(temp_file_path)

    return json_data


def build_call_graph(json_data, base_dirs, python_files):
    G = nx.DiGraph()

    for node_id, node_data in json_data["graph"]["nodes"].items():
        G.add_node(node_id, **node_data)

    for edge in json_data["graph"]["edges"]:
        G.add_edge(edge["source"], edge["target"])

    return G


def get_function_source(file_path, function_name):
    with open(file_path, "r") as f:
        content = f.read()

    # Parse the content into an AST
    atok = asttokens.ASTTokens(content, parse=True)

    # Find the function definition
    for node in ast.walk(atok.tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Get the source code for the function
            return atok.get_text(node)

    # If the function is not found, return an empty string
    return ""


def gather_function_code(G, start_function, base_dirs):
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

        code += get_function_source(relative_path, function_name) + "\n\n"

        for neighbor in G.successors(node_id):
            if neighbor not in visited:
                stack.append(neighbor)

    return code


def hash_function(code):
    return hashlib.sha256(code.encode()).hexdigest()


def find_node_by_name(G, name):
    for node, data in G.nodes(data=True):
        print(data["name"])
        if data["name"] == name:
            return node
    return None


def format_module_path(module_string):
    # Split the string at the last period
    module_parts = module_string.rsplit(".", 1)

    if len(module_parts) == 1:
        # If there's no period, just add .py at the end
        return f"{module_string}.py"

    # Replace periods with slashes in the module path
    file_path = module_parts[0].replace(".", "/")

    # Combine the path, add .py, and append the function name
    return f"dcft/{file_path}.py::{module_parts[1]}"


def hash_function(start_function_name: str):
    start_function_name = format_module_path(start_function_name)
    start_function_name
    base_dirs = ["dcft/data_strategies", "dcft/external_repositories"]

    python_files = get_python_files(base_dirs)
    output_file = run_code2flow(python_files)
    json_data = parse_code2flow_output(output_file)

    call_graph = build_call_graph(json_data, base_dirs, python_files)

    start_function_name = "dcft/data_strategies/WizardLM/utils.py::annotate"
    start_node = find_node_by_name(call_graph, start_function_name)

    if start_node is None:
        print(f"Error: Function '{start_function_name}' not found in the call graph.")
        return

    function_code = gather_function_code(call_graph, start_node, base_dirs)
    function_hash = hash_function(function_code)

    print(f"Hash for function '{start_function_name}': {function_hash}")
