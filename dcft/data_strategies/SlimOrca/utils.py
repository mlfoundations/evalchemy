import random
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset, load_dataset
import logging

ORCA_SYSTEM_PROMPTS = [
    "",
    "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
    "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old.",
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.",
    "You are an AI assistant that helps people find information. Provide a detailed answer so user don't need to search outside to understand the answer",
    "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",
    "You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. Think like you are answering to a five year old.",
    "Explain how you used the definition to come up with the answer.",
    "You are an AI assistant. You should describe the task and explain your answer. While answering a multiple choice question, first output the correct answer(s). Then explain why other answers are wrong. You might need to use additional knowledge to answer the question.",
    "You are an AI assistant that helps people find information. User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-bystep and justify your answer.",
    "User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.",
    "You are a teacher. Given a task, you explain in simple steps what the task is asking, any guidelines it provides and how to use those guidelines to find the answer.",
    "You are an AI assistant, who knows every language and how to translate one language to another. Given a task, you explain in simple steps what the task is asking, any guidelines that it provides. You solve the task and show how you used the guidelines to solve the task.",
    """Given a definition of a task and a sample input, break the definition into small parts.
    Each of those parts will have some instruction. Explain their meaning by showing an
    example that meets the criteria in the instruction. Use the following format:
    Part #: a key part of the definition.
    Usage: Sample response that meets the criteria from the key part. Explain why you think
    it meets the criteria.""",
    "You are an AI assistant that helps people find information.",
]

MAPPING_ORCA_TASK_INDEX = {
    "cot": [5, 10, 15],
    "niv2": [0, 1, 4, 6, 8, 11, 12, 13, 14],
    "t0": [0, 1, 2, 4, 6],
    "flan": [2, 3, 6, 7, 9],
}


def add_system_instructions(dataset: Dataset, mixture_name: str, instruction_column: str = "inputs") -> Dataset:
    """Add system instructions to each instance in the dataset based on the mixture type.

    Args:
        dataset (Dataset): Input dataset to augment
        mixture_name (str): Name of the mixture type (cot, niv2, t0, or flan)
        instruction_column (str, optional): Column containing instructions. Defaults to "inputs"

    Returns:
        Dataset: Dataset with added system instructions
    """
    indices = MAPPING_ORCA_TASK_INDEX[mixture_name]
    system_prompts = [ORCA_SYSTEM_PROMPTS[index] for index in indices]

    def add_system_instruction_to_instance(instance):
        # if mixture_name == "flan" and ("options:" not in instance[instruction_column].lower()):
        #     # don't use instructions 8,10 (index 7 and 9) for flan questions that are not multiple choice
        #     sampled_system_prompt = random.choice(system_prompts[:-2])
        # else:
        #     sampled_system_prompt = random.choice(system_prompts)

        # NOTE: becuase we are currently only using flan zsopt data, we are assuming all the questions are multiple choice (which may not be true)
        # In the original paper they only apply instruction 7,9 to those with multiple choices. They (maybe) are sampling from both flan zsopt and flan zsnoopt
        sampled_system_prompt = random.choice(system_prompts)
        instance["system_instruction"] = sampled_system_prompt
        return instance

    # There is a bug that shows up for very large datasets when num_proc is used. https://github.com/huggingface/datasets/issues/6393
    # dataset = dataset.map(add_system_instruction_to_instance, desc="Adding system instructions", num_proc=cpu_count())
    # can try sharding this part?
    dataset = dataset.map(add_system_instruction_to_instance, desc="Adding system instructions")

    return dataset


def get_task_to_indices_map(dataset: Dataset, shuffle: bool = False) -> dict:
    """Create a mapping of task names to their corresponding indices in the dataset.

    Args:
        dataset (Dataset): Input dataset
        shuffle (bool, optional): Whether to shuffle indices for each task. Defaults to False

    Returns:
        dict: Mapping of task names to lists of indices
    """
    task_to_indices_map = defaultdict(list)
    for i, task in tqdm(
        enumerate(dataset["_task_name"]), total=len(dataset["_task_name"]), desc="Mapping tasks to indices"
    ):
        task_to_indices_map[task].append(i)

    if shuffle:
        for indices in task_to_indices_map.values():
            random.shuffle(indices)

    print(f"Number of tasks in {dataset.info.dataset_name}: {len(task_to_indices_map)}")
    return task_to_indices_map


def load_and_uniform_sample_dataset(dataset_name: str, num_samples: int, data_dir: str, seed: int = 314) -> Dataset:
    """Load a dataset and uniformly sample a fixed number of instances.

    Args:
        dataset_name (str): Name of the dataset to load
        num_samples (int): Number of samples to select
        seed (int, optional): Random seed. Defaults to 314

    Returns:
        Dataset: Sampled dataset
    """
    if data_dir:
        dataset = load_dataset(dataset_name, data_dir=data_dir, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    return dataset.shuffle(seed=seed).select(range(num_samples))


def load_and_sample_dataset(
    dataset_name: str, data_dir: str, strategy: str, num_queries: int, seed: int = 314
) -> Dataset:
    """Load a dataset and sample according to specified strategy.

    Args:
        dataset_name (str): Name of the dataset to load
        data_dir (str): Directory containing the dataset
        strategy (str): Sampling strategy ('num_queries_per_task' or 'total_num_queries_stratified')
        num_queries (int): Number of queries to sample (interpretation depends on strategy)
        seed (int, optional): Random seed. Defaults to 314

    Returns:
        Dataset: Sampled dataset
    """
    dataset = load_dataset(dataset_name, data_dir=data_dir, split="train")
    dataset.info.dataset_name = dataset_name + "/" + data_dir
    if strategy == "num_queries_per_task":
        return sample_num_queries_per_task(dataset, num_queries_per_task=num_queries, seed=seed)
    elif strategy == "total_num_queries_stratified":
        return sample_total_num_queries_stratified(dataset, total_num_queries=num_queries, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def load_and_filter_dataset(dataset_name: str, filter_column: str) -> Dataset:
    """Load a dataset and filter out instances with falsey values in specified column.

    Args:
        dataset_name (str): Name of the dataset to load
        filter_column (str): Column to use for filtering

    Returns:
        Dataset: Filtered dataset
    """
    dataset = load_dataset(dataset_name, split="train")
    num_before = len(dataset)
    # filters out python Falsey values (e.g. 0, 0.0, False, None, "", etc.)
    dataset = dataset.filter(lambda x: x[filter_column])
    num_after = len(dataset)
    percentage_removed = (num_before - num_after) / num_before * 100
    logging.info(
        f"Filtered {num_before - num_after} out of {num_before} samples for {dataset_name}, removing ({percentage_removed:.2f}%)"
    )
    return dataset


def sample_num_queries_per_task(dataset: Dataset, num_queries_per_task: int, seed: int = 314):
    """Sample a fixed number of queries from each task in the dataset.

    Args:
        dataset (Dataset): Input dataset
        num_queries_per_task (int): Number of queries to sample per task
        seed (int, optional): Random seed. Defaults to 314

    Returns:
        Dataset: Sampled dataset
    """
    random.seed(seed)
    task_to_indices_map = get_task_to_indices_map(dataset)

    print(f"Sampling {num_queries_per_task} queries per task for {dataset.info.dataset_name}")

    sampled_indices = []
    for task, indices in tqdm(task_to_indices_map.items(), desc="Sampling queries per task"):
        if len(indices) > num_queries_per_task:
            sampled_indices.extend(random.sample(indices, num_queries_per_task))
        else:
            print(f"WARNING: {task} has only {len(indices)} of requested {num_queries_per_task} queries. Sampling all.")
            sampled_indices.extend(indices)

    print(f"Sampled {len(sampled_indices)} total queries for {dataset.info.dataset_name}")

    return dataset.select(sampled_indices)


# See Algorithm 1 in https://arxiv.org/pdf/2306.02707
def sample_total_num_queries_stratified(dataset: Dataset, total_num_queries: int, seed: int = 314):
    """Sample queries using stratified sampling across tasks.

    Args:
        dataset (Dataset): Input dataset
        total_num_queries (int): Total number of queries to sample
        seed (int, optional): Random seed. Defaults to 314

    Returns:
        Dataset: Sampled dataset
    """
    random.seed(seed)
    # instead of randomly sampling from the list of indices again and again, can simply just shuffle before and then pop() (which is O(1) if popping the last)
    task_to_indices_map = get_task_to_indices_map(dataset, shuffle=True)
    sampled_indices = []
    non_empty_tasks = set(task_to_indices_map.keys())

    while len(sampled_indices) < total_num_queries and non_empty_tasks:
        task = random.choice(list(non_empty_tasks))
        indices = task_to_indices_map[task]

        if indices:
            index = indices.pop()
            sampled_indices.append(index)
        else:
            non_empty_tasks.remove(task)

    print(f"Sampled {len(sampled_indices)} total queries for {dataset.info.dataset_name}")
    return dataset.select(sampled_indices)
