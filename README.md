# How to run existing evaluation frameworks?

We adopt a lot from the amazing LM-Eval-Harness [https://github.com/EleutherAI/lm-evaluation-harness]. We accept the whole model interface from lm-eval-harness.

Evaluation is as simple as 
Note: Please make sure OPENAI_API_KEY is set!
```bash
python -m eval.eval 
    --model hf  \
    --tasks HumanEval \
    --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size auto \
    --output_path logs
```

The output will be placed in the logs file after.

The list of tasks is a comma separated list of tasks. The list of instruction based tasks can be seen under "eval/chat_benchmarks". All pretrained tasks in [LM-Eval-Harness repository](https://github.com/EleutherAI/lm-evaluation-harness) are also available in this framework. To utilize data-parallelism (faster and what we recommend), we suggest using accelerate 

```bash
accelerate launch --num-processes <num-gpus> --num-machines <num-nodes> \
    --multi-gpu -m eval.eval \
    --model hf \
    --task MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Llama-3.1-8B-Instruct' \
    --batch_size 2 \
    --output_path logs
```

If your model cannot fit on one single GPU, we suggest using model-parallelism (slower and less recommended)
```bash
python -m eval.eval \
    --model hf \
    --task MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Llama-3.1-8B-Instruct,parallelize=True' \
    --batch_size 2 \
    --output_path logs
```
While we do support "auto" batch_size, we do recommend playing with the batch_size yourself as it is fairly conservative. 

## Annotator changes
To change the annotator for all judges, please use 
```bash
    --annotator_model gpt-4o
```

## Logging to the leaderboard

We also support automatically logging to our leaderboard!

```bash
python -m eval.eval \
    --model hf \
    --task MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Llama-3.1-8B-Instruct,parallelize=True' \
    --batch_size 2 \
    --output_path logs \
    --use-database
```




You can optionally pass in the following flags to add information to your uploaded result: 

```bash
    --model_name <name of your model in database> \
    --creation_location <where this model was created> \
    --created_by <who created this model> \
```

The model will appear here at the [leaderboard](https://llm-leaderboard-319533213591.us-central1.run.app/).

To log into this database, please set this environmental variables
```bash
export DB_PASSWORD='t}LQ7ZL]3$x~I8ye'
export DB_HOST='35.225.163.235'
export DB_PORT='5432'
export DB_NAME="postgres"
export DB_USER='postgres'
```

# Installation instructions

To install, please follow the following steps:
```bash
conda create --name dcft python=3.10
conda activate dcft      
pip install -e ".[eval]"
pip install -r requirements.txt
```

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

outputs = model.generate_until(all_instances)
```

2. Try to use lm-eval logger as much as possible.