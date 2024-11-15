# DCFT Evaluation Framework

This evaluation framework builds upon the [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness) and provides additional capabilities for evaluating language models. The framework supports both standard benchmarks and custom evaluation tasks.

## Quick Start

### Installation

```bash
# Create and activate conda environment
conda create --name dcft python=3.10
conda activate dcft      

# Install dependencies
pip install -e ".[eval]"
pip install -r requirements.txt
```

### Basic Usage

Make sure your `OPENAI_API_KEY` is set in your environment before running evaluations.

```bash
python -m eval.eval \
    --model hf \
    --tasks HumanEval \
    --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size auto \
    --output_path logs
```

## Advanced Usage

### Multi-GPU Evaluation

For faster evaluation using data parallelism (recommended):

```bash
accelerate launch --num-processes <num-gpus> --num-machines <num-nodes> \
    --multi-gpu -m eval.eval \
    --model hf \
    --tasks MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Llama-3-8B-Instruct' \
    --batch_size 2 \
    --output_path logs
```

### Large Model Evaluation

For models that don't fit on a single GPU, use model parallelism:

```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Llama-3-8B-Instruct,parallelize=True' \
    --batch_size 2 \
    --output_path logs
```

> **Note**: While "auto" batch size is supported, we recommend manually tuning the batch size for optimal performance.

### Customizing Evaluation

#### Change Annotator Model
```bash
--annotator_model gpt-4o-mini-2024-07-18
```

## Leaderboard Integration

### Database Configuration

Set the following environment variables to enable database logging:

```bash
export DB_PASSWORD=<DB_PASSWORD>
export DB_HOST=<DB_HOST>
export DB_PORT=<DB_PORT>
export DB_NAME=<DB_NAME>
export DB_USER=<DB_USER>
```

### Submit Results to Leaderboard

```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Llama-3-8B-Instruct' \
    --batch_size 2 \
    --output_path logs \
    --use-database \
    --model_name "My Model Name" \
    --creation_location "Lab Name" \
    --created_by "Researcher Name"
```

View results on the [leaderboard](https://llm-leaderboard-319533213591.us-central1.run.app/).

### Updating Database Results

You can update existing results using either:

1. Model ID: `--model_id <YOUR_MODEL_ID>`
2. Model Name: `--model-name <MODEL_NAME_IN_DB>`

Note: If both are provided, model_id takes precedence.

## Implementing Custom Evaluations

To add a new evaluation system:

1. Create a new directory under `eval/chat_benchmarks/`
2. Implement `eval_instruct.py` with two required functions:
   - `eval_instruct(model)`: Takes an LM Eval Model, returns results dict
   - `evaluate(results)`: Takes results dict, returns evaluation metrics

### Adding External Evaluation Repositories

Use git subtree to manage external evaluation code:

```bash
# Add external repository
git subtree add --prefix=eval/chat_benchmarks/new_eval https://github.com/original/repo.git main --squash

# Pull updates
git subtree pull --prefix=eval/chat_benchmarks/new_eval https://github.com/original/repo.git main --squash

# Push contributions back
git subtree push --prefix=eval/chat_benchmarks/new_eval https://github.com/original/repo.git contribution-branch
```

### Debug Mode

To run evaluations in debug mode, add the `--debug` flag:

```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench \
    --model_args "pretrained=meta-llama/Llama-3-8B-Instruct" \
    --batch_size 2 \
    --output_path logs \
    --debug
```

Debug mode provides:
This is particularly useful when testing new evaluation implementations, debugging model configurations, verifying dataset access, and testing database connectivity.

### Performance Tips

1. Utilize batch processing for faster evaluation:
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

2. Use the LM-eval logger for consistent logging across evaluations

## Available Tasks

### Built-in Benchmarks
- All tasks from [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Custom instruction-based tasks (found in `eval/chat_benchmarks/`):
  - **MTBench**: Multi-turn dialogue evaluation benchmark for assessing chat capabilities
  - **WildBench**: Real-world task evaluation across diverse domains and scenarios
  - **RepoBench**: Code understanding and repository-level programming tasks
  - **MixEval**: Comprehensive evaluation across multiple domains including reasoning, math, and coding
  - **AlpacaEval**: Instruction following evaluation
  - **HumanEval**: Code generation and problem solving
  - **ZeroEval**: Logical reasoning and problem solving
  - **MBPP**: Python programming benchmark

Example running multiple benchmarks:
```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,WildBench,RepoBench,MixEval \
    --model_args "pretrained=meta-llama/Llama-3-8B-Instruct" \
    --batch_size auto \
    --output_path logs
```

### Special Access Requirements

#### ZeroEval Access
To run ZeroEval benchmarks, you need to:

1. Request access to the [ZebraLogicBench-private dataset](https://huggingface.co/datasets/allenai/ZebraLogicBench-private) on Hugging Face
2. Accept the terms and conditions
3. Log in to your Hugging Face account when running evaluations
