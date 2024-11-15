from datasets import Dataset
from typing import List


def convert_to_sharegpt_format(conversations: List) -> List:
    """
    Convert a list of conversation dictionaries to ShareGPT format.

    Args:
        conversations (list): List of dictionaries containing 'role' and 'content'

    Returns:
        dict: Formatted conversation in ShareGPT format
    """
    formatted_conversations = []

    for message in conversations:
        from_role = "human" if message["role"] == "user" else "gpt"
        formatted_message = {"from": from_role, "value": message["content"]}
        formatted_conversations.append(formatted_message)

    return formatted_conversations


def convert(dataset: Dataset) -> Dataset:
    def filter(ex):

        ex["conversations"] = convert_to_sharegpt_format(ex["chosen"])[:-1]
        ex["chosen"] = convert_to_sharegpt_format(ex["chosen"])[-1]
        ex["rejected"] = convert_to_sharegpt_format(ex["rejected"])[-1]
        return ex

    dataset = dataset.map(filter)
    return dataset
