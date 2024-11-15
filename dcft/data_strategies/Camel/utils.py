import random

from typing import List
from tqdm import tqdm
from datasets import Dataset

from dcft.external_repositories.camel.examples.chemistry.task_generation import (
    generate_tasks as generate_chemistry_tasks,
)
from dcft.external_repositories.camel.examples.physics.task_generation import generate_tasks as generate_physics_tasks
from dcft.external_repositories.camel.examples.math.task_generation import generate_tasks as generate_math_tasks
from dcft.external_repositories.camel.examples.biology.task_generation import generate_tasks as generate_biology_tasks
from camel.types import ModelType
from dcft.external_repositories.camel.examples.math.role_playing_multiprocess import (
    generate_examples as generate_math_examples,
)
from dcft.external_repositories.camel.examples.physics.role_playing_multiprocess import (
    generate_examples as generate_physics_examples,
)
from dcft.external_repositories.camel.examples.chemistry.role_playing_multiprocess import (
    generate_examples as generate_chemistry_examples,
)
from dcft.external_repositories.camel.examples.biology.role_playing_multiprocess import (
    generate_examples as generate_biology_examples,
)


# MODEL_TYPE=ModelType.GPT_4O
MODEL_TYPE = ModelType.GPT_4O_MINI


def generate_math_topic(num_topics: int = 25, num_tasks_per_topic: int = 25) -> Dataset:
    """
    Generate topics and subtopics for math

    Returns:
        Dataset: A dataset with a new column for topic and sub_topic
    """
    dataset = generate_math_tasks(model_type=MODEL_TYPE, num_topics=num_topics, num_tasks_per_topic=num_tasks_per_topic)
    return dataset


def generate_math_qa(
    dataset: Dataset,
    num_problems_per_task: int = 80,
    num_topics: int = 25,
    num_tasks_per_topic: int = 25,
    max_samples: int = 50_000,
) -> Dataset:
    """
    Generate question answer pairs for math

    Returns:
        Dataset: A dataset with new columns for question and answer pairs
    """
    dataset = generate_math_examples(
        dataset,
        model_type=MODEL_TYPE,
        num_problems_per_task=num_problems_per_task,
        num_topics=num_topics,
        num_tasks_per_topic=num_tasks_per_topic,
        max_samples=max_samples,
    )
    return dataset


def generate_physics_topic(num_topics: int = 25, num_tasks_per_topic: int = 25) -> Dataset:
    """
    Generate topics and subtopics for physics

    Returns:
        Dataset: A dataset with a new column for topic and sub_topic
    """
    dataset = generate_physics_tasks(
        model_type=MODEL_TYPE, num_topics=num_topics, num_tasks_per_topic=num_tasks_per_topic
    )
    return dataset


def generate_physics_qa(
    dataset: Dataset,
    num_problems_per_task: int = 32,
    num_topics: int = 25,
    num_tasks_per_topic: int = 25,
    max_samples: int = 20_000,
) -> Dataset:
    """
    Generate question answer pairs for physics

    Returns:
        Dataset: A dataset with new columns for question and answer pairs
    """
    dataset = generate_physics_examples(
        dataset,
        model_type=MODEL_TYPE,
        num_problems_per_task=num_problems_per_task,
        num_topics=num_topics,
        num_tasks_per_topic=num_tasks_per_topic,
        max_samples=max_samples,
    )
    return dataset


def generate_biology_topic(num_topics: int = 25, num_tasks_per_topic: int = 25) -> Dataset:
    """
    Generate topics and subtopics for biology

    Returns:
        Dataset: A dataset with a new column for topic and sub_topic
    """
    dataset = generate_biology_tasks(
        model_type=MODEL_TYPE, num_topics=num_topics, num_tasks_per_topic=num_tasks_per_topic
    )
    return dataset


def generate_biology_qa(
    dataset: Dataset,
    num_problems_per_task: int = 32,
    num_topics: int = 25,
    num_tasks_per_topic: int = 25,
    max_samples: int = 20_000,
) -> Dataset:
    """
    Generate question answer pairs for biology

    Returns:
        Dataset: A dataset with new columns for question and answer pairs
    """
    dataset = generate_biology_examples(
        dataset,
        model_type=MODEL_TYPE,
        num_problems_per_task=num_problems_per_task,
        num_topics=num_topics,
        num_tasks_per_topic=num_tasks_per_topic,
        max_samples=max_samples,
    )
    return dataset


def generate_chemistry_topic(num_topics: int = 25, num_tasks_per_topic: int = 25) -> Dataset:
    """
    Generate topics and subtopics for chemistry

    Returns:
        Dataset: A dataset with a new column for topic and sub_topic
    """
    dataset = generate_chemistry_tasks(
        model_type=MODEL_TYPE, num_topics=num_topics, num_tasks_per_topic=num_tasks_per_topic
    )
    return dataset


def generate_chemistry_qa(
    dataset: Dataset,
    num_problems_per_task: int = 32,
    num_topics: int = 25,
    num_tasks_per_topic: int = 25,
    max_samples: int = 20_000,
) -> Dataset:
    """
    Generate question answer pairs for chemistry

    Returns:
        Dataset: A dataset with new columns for question and answer pairs
    """
    dataset = generate_chemistry_examples(
        dataset,
        model_type=MODEL_TYPE,
        num_problems_per_task=num_problems_per_task,
        num_topics=num_topics,
        num_tasks_per_topic=num_tasks_per_topic,
        max_samples=max_samples,
    )
    return dataset
