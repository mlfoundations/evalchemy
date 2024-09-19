import json
import random
import datasets
from tqdm import tqdm

from dcft.external_repositories.WizardLM.Evol_Instruct.openai_access import call_chatgpt
from dcft.external_repositories.WizardLM.Evol_Instruct.depth import (
    createConstraintsPrompt,
    createDeepenPrompt,
    createConcretizingPrompt,
    createReasoningPrompt,
)
from dcft.external_repositories.WizardLM.Evol_Instruct.breadth import createBreadthPrompt


def instruction_generation(inputs):
    inputs = inputs[:3]
    evol_instructions = []
    for instruction in tqdm(inputs):
        evol_prompts = []
        evol_prompts.append(createConstraintsPrompt(instruction))
        evol_prompts.append(createDeepenPrompt(instruction))
        evol_prompts.append(createConcretizingPrompt(instruction))
        evol_prompts.append(createReasoningPrompt(instruction))
        evol_prompts.append(createBreadthPrompt(instruction))

        selected_evol_prompt = random.choice(evol_prompts)

        evol_instruction = call_chatgpt(selected_evol_prompt)
        evol_instructions.append(evol_instruction)

    return evol_instructions


def annotation_generation(instructions):
    pairs = []
    for instruction in tqdm(instructions):
        pairs.append({"instruction": instruction, "output": call_chatgpt(instruction)})
    return pairs
