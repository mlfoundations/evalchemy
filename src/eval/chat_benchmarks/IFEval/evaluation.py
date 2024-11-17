import logging
import collections
import dataclasses
import json

from .evaluation_main import (
    test_instruction_following_strict,
    test_instruction_following_loose,
    read_prompt_to_response_dict,
    read_prompt_list,
)


def get_report(outputs):
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = collections.defaultdict(int)
    tier0_correct = collections.defaultdict(int)

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            instruction_id = instruction_id.split(":")[0]
            tier0_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[instruction_id] += 1

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1

    for instruction_id in sorted(tier0_total.keys()):
        tight_accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
        logging.info(f"tier0 accuracy {instruction_id} {tight_accuracy}")

    for instruction_id in sorted(tier1_total.keys()):
        loose_accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
        logging.info(f"tier1 accuracy {instruction_id} {loose_accuracy}")

    return {"prompt-level": prompt_correct / prompt_total, "instruction-level": instruction_correct / instruction_total}


def evaluate_accuracy(response_filename):
    inputs = read_prompt_list(response_filename)
    prompt_to_response = read_prompt_to_response_dict(response_filename)

    for func, output_file in [
        (test_instruction_following_strict, "eval_results_strict"),
        (test_instruction_following_loose, "eval_restuls_loose"),
    ]:
        # logging.info(f"Generating {output_file}")
        outputs = []
        for inp in inputs:
            outputs.append(func(inp, prompt_to_response))

        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)

        logging.info(f"Accuracy: {accuracy}")

    return get_report(outputs)
