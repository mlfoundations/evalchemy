# How to run existing evaluation frameworks?

We adopt a lot from the amazing LM-Eval-Harness [https://github.com/EleutherAI/lm-evaluation-harness]. We accept the whole model interface from lm-eval-harness.

Evaluation is as simple as 

```bash
python -m eval.eval --model hf  --tasks HumanEval --model_args 'pretrained=meta-llama/Meta-Llama-3-8B-Instruct' --batch_size auto --output_path logs
```

The "auto" batch-size significantly improves speed. The output will be placed in the logs file. 

### Database Updates for Evaluation Results

You can update the database using either:
1. Search by Model ID
Use the model's unique identifier: ``` --model_id <YOUR_MODEL_ID> ```

2. Search by Model Name
To search using the model's name instead: ``` --model-name <MODEL_NAME_IN_DB> ```

If both model_name and model_id are supplied, then model_id will take precedence.

## Implementing a new evaluation system

1. Add the relevant repository with code for your evaluation system in a folder under eval/chat_benchmarks. If it is a git repository that is maintained, I recommend using 
```bash
    git subtree add --prefix=eval/chat_benchmarks/alpaca_eval https://github.com/original/repo.git main --squash

    # Make changes in the eval/chat_benchmarks/alpaca_eval directory

    # Commit changes to your main repository
    git add eval/chat_benchmarks/alpaca_eval
    git commit -m "Update library-name with custom changes"

    # To pull updates from the original repository
    git subtree pull --prefix=eval/chat_benchmarks/alpaca_eval https://github.com/original/repo.git main --squash

    # If you want to contribute back to the original repository
    git subtree push --prefix=eval/chat_benchmarks/alpaca_eval https://github.com/original/repo.git contribution-branch
```


2. Inside the folder, please create a file eval_instruct.py that has two functions: 1. eval_instruct(model) which takes a LM Eval Model and outputs a results dict and 2. evaluate which takes in results dict and outputs another results dict with the evaluation results. For example, see eval/chat_benchmarks/MBPP/eval_instruct.py

It's that easy. 

Some tips to make this faster. 

1. Utilizing the automatic batching from lm-eval-harness is best for improving speed. Put all the generation instances into a list and then pass this off to lm-eval-harness and it will generate all. 
```python
all_instances.append(
        Instance(
            "generate_until",
            example,
            (
                inputs,
                {
                    "max_new_tokens": 1024,
                    "do_sample": False,
                },
            ),
            idx,
        )
)

outputs = self.compute(model, all_instances)
```

2. Try to use lm-eval logger as much as possible.