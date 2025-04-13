"""
The logic in this file largely borrows from Qwen2.5-Math codebase at https://github.com/QwenLM/Qwen2.5-Math:
"""

import re


def get_multiple_choice_answer(pred: str):
    """
    Extract the multiple choice answer from the prediction string.
    HLE multiple-choice questions involve one of five or more answer choices.
    Model is prompted to output answer after the "Answer:" string.
    """

    tmp = re.findall(r"Answer:\s*([A-Z])", pred)

    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred
