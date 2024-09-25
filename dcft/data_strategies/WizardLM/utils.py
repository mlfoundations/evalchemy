import random

from typing import List
from tqdm import tqdm
from datasets import Dataset

from dcft.external_repositories.WizardLM.Evol_Instruct.openai_access import call_chatgpt
from dcft.external_repositories.WizardLM.Evol_Instruct.depth import (
    createConstraintsPrompt,
    createDeepenPrompt,
    createConcretizingPrompt,
    createReasoningPrompt,
)
from dcft.external_repositories.WizardLM.Evol_Instruct.breadth import createBreadthPrompt


def instruction_generation(dataset: Dataset, input_column: str, output_column: str) -> Dataset:
    """
    Generate evolved instructions for each input in the dataset.

    Args:
        dataset (Dataset): The input dataset.
        input_column (str): The name of the column containing input instructions.
        output_column (str): The name of the column to store generated instructions.

    Returns:
        Dataset: The dataset with a new column containing evolved instructions.
    """
    evol_instructions: List[str] = []
    inputs = dataset[input_column]
    for instruction in tqdm(inputs):
        evol_prompts: List[str] = []
        evol_prompts.append(createConstraintsPrompt(instruction))
        evol_prompts.append(createDeepenPrompt(instruction))
        evol_prompts.append(createConcretizingPrompt(instruction))
        evol_prompts.append(createReasoningPrompt(instruction))
        evol_prompts.append(createBreadthPrompt(instruction))

        selected_evol_prompt: str = random.choice(evol_prompts)

        evol_instruction: str = call_chatgpt(selected_evol_prompt)
        evol_instructions.append(evol_instruction)

    dataset = dataset.add_column(output_column, evol_instructions)
    return dataset


def annotate(dataset: Dataset, input_column: str, output_column: str) -> Dataset:
    """
    Annotate each input in the dataset using ChatGPT.

    Args:
        dataset (Dataset): The input dataset.
        input_column (str): The name of the column containing inputs to annotate.
        output_column (str): The name of the column to store annotations.

    Returns:
        Dataset: The dataset with a new column containing annotations.
    """
    inputs = dataset[input_column]
    annotations = []
    for input in tqdm(inputs):
        annotations.append(call_chatgpt(input))
    dataset = dataset.add_column(output_column, annotations)
    return dataset


def dedup(dataset: Dataset, input_column: str) -> Dataset:
    """
    Remove duplicate rows from the dataset based on a specific column.

    Args:
        dataset (Dataset): The input dataset.
        input_column (str): The name of the column to check for duplicates.

    Returns:
        Dataset: The dataset with duplicate rows removed.
    """
    # Convert to pandas DataFrame
    df = dataset.to_pandas()

    # Drop duplicate rows based on the specified column
    df_cleaned = df.drop_duplicates(subset=[input_column], keep="first")

    # Convert back to Hugging Face Dataset
    cleaned_dataset = Dataset.from_pandas(df_cleaned)

    return cleaned_dataset


def force_rename_column(dataset: Dataset, old_name: str, new_name: str) -> Dataset:
    """
    Rename a column in the dataset, removing any existing column with the new name.

    Args:
        dataset (Dataset): The input dataset.
        old_name (str): The current name of the column to be renamed.
        new_name (str): The new name for the column.

    Returns:
        Dataset: The dataset with the renamed column.
    """
    column_names = dataset.column_names

    if new_name in column_names:
        dataset = dataset.remove_columns(new_name)

    name_mapping = {old_name: new_name}
    renamed_dataset = dataset.rename_columns(name_mapping)

    return renamed_dataset


def remove_other_columns(dataset: Dataset, columns_to_keep: List[str]) -> Dataset:
    all_columns = dataset.column_names
    columns_to_remove = [col for col in all_columns if col not in columns_to_keep]
    cleaned_dataset = dataset.remove_columns(columns_to_remove)
    return cleaned_dataset
