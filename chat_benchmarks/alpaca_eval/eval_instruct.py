from alpaca_eval.main import evaluate as alpaca_eval_evaluate
from lm_eval.api.instance import Instance
import torch
import datasets 
from tqdm import tqdm

def eval_instruct(model):
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    outputs = []
    with torch.no_grad():
        all_instances = []
        for idx, example in enumerate(tqdm(eval_set)):
            instruction = example['instruction']
            instruction = model.apply_chat_template([{"role": "user", "content": instruction}])

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        instruction,
                        {
                            "max_new_tokens": 1024,
                            "do_sample": True,
                            "temperature": 0.5
                        },
                    ),
                    idx,
                )
            )
            
        outputs = model.generate_until(all_instances)
        model_outputs = []
        for idx, example in enumerate(tqdm(eval_set)):
            instruction = example['instruction']
            instance = {}
            instance['instruction'] = instruction
            instance['dataset'] = example['dataset']
            instance['datasplit'] = "eval"
            instance['generator'] = model.model_identifier
            instance['output'] = outputs[idx]
            model_outputs.append(instance)
            
    results = {}
    results['model_outputs'] = model_outputs
    results['model_identifier'] = model.model_identifier
    return results

def evaluate(results):
    model_outputs = results['model_outputs']
    leaderboard = alpaca_eval_evaluate(model_outputs=model_outputs, is_return_instead_of_print=True)
    return leaderboard[0].loc[results['model_identifier']].to_dict()