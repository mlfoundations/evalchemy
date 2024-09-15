from alpaca_eval.main import evaluate as alpaca_eval_evaluate
from lm_eval.api.instance import Instance

def eval_instruct(model):
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    outputs = []
    with torch.no_grad():
        all_instances = []
        for example in tqdm(eval_set):
            instruction = example['instruction']
            

            all_instances.append(
                Instance(
                    "generate_until",
                    instance,
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
        for idx, output in enumerate(outputs):
            instance = {}
            instance['instruction'] = instruction
            instance['generator'] = "model"
            instance['dataset'] = example['dataset']
            instance['datasplit'] = "eval"

            instance['output'] = output
            model_outputs.append(instance)
        results['model_outputs'] = model_outputs
