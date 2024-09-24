import json
from typing import List, Dict, Any
import random
import datasets
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


def instruction_generation(dataset: Dataset, input_column: str, output_column:str) -> Dataset:
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
    return evol_instructions


def annotate(dataset: Dataset, input_column: str, output_column:str) -> Dataset:
    inputs = dataset[input_column]
    annotations = []
    for input in tqdm(inputs):
        annotations.append(call_chatgpt(input))
    dataset = dataset.add_column(output_column, annotations)
    return dataset

def dedup(dataset: Dataset, input_column: str) -> Dataset:
    return dataset.unique(input_column)
