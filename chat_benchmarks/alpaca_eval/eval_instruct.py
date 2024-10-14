from typing import Dict, List, Any
from alpaca_eval.main import evaluate as alpaca_eval_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import torch
import datasets
from tqdm import tqdm


def eval_instruct(model: LM) -> Dict[str, Any]:
    """
    Generate the completions for the model on the Alpaca dataset.

    Args:
        model (Any): The language model to evaluate.

    Returns:
        Dict[str, Any]: A dictionary containing the generations of the model
        including model outputs and model identifier.
    """
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    outputs = []
    with torch.no_grad():
        all_instances: List[Instance] = []
        for idx, example in enumerate(tqdm(eval_set)):
            instruction = example["instruction"]
            instruction = model.apply_chat_template([{"role": "user", "content": instruction}])

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        instruction,
                        {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.5},
                    ),
                    idx,
                )
            )

        outputs = model.generate_until(all_instances)

        model_outputs: List[Dict[str, Any]] = []
        for idx, example in enumerate(tqdm(eval_set)):
            instruction = example["instruction"]
            instance = {
                "instruction": instruction,
                "dataset": example["dataset"],
                "datasplit": "eval",
                "generator": model.model_identifier,
                "output": outputs[idx],
            }
            model_outputs.append(instance)

    results = {"model_outputs": model_outputs, "model_identifier": model.model_identifier}
    return results


def evaluate(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluate the model outputs using the Alpaca Eval framework.

    Args:
        results (Dict[str, Any]): A dictionary containing model outputs and identifier.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics for the model.
    """
    model_outputs = results["model_outputs"]
    leaderboard = alpaca_eval_evaluate(model_outputs=model_outputs, is_return_instead_of_print=True)
    return leaderboard[0].loc[results["model_identifier"]].to_dict()
