# 🧪 Evalchemy

> *A framework for Gold Standard Language Model Evaluations*

This evaluation framework builds upon the [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness) to provide a unified, easy-to-use platform for language model evaluation. We've streamlined the process by:

- Integrating multiple popular evaluation repositories into a single, cohesive framework.
- Providing simple installation and unified dependencies.
- Supporting both data-parallel and model-parallel evaluation strategies.
- Offering consistent interfaces across different benchmarks.

### Key Features

- **Unified Installation**: One-step setup for all benchmarks, eliminating dependency conflicts
- **Parallel Evaluation**:
  - Data-Parallel: Distribute evaluations across multiple GPUs for faster results
  - Model-Parallel: Handle large models that don't fit on a single GPU
- **Simplified Usage**: Run any benchmark with consistent command-line interface

Additional Features:
- **Results Management**: 
  - Local results tracking with standardized output format
  - Optional database integration for systematic tracking
  - Leaderboard submission capability (requires database setup)

## ⚡ Quick Start

### Installation

```bash
# Create and activate conda environment
conda create --name evalchemy python=3.10
conda activate evalchemy      

# Install dependencies
pip install -e ".[eval]"

# Log into HuggingFace for datasets and models.
huggingface-cli login
```

## 📚 Available Tasks

### Built-in Benchmarks
- All tasks from [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Custom instruction-based tasks (found in `eval/chat_benchmarks/`):
  - **MTBench**: [Multi-turn dialogue evaluation benchmark](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
  - **WildBench**: [Real-world task evaluation](https://github.com/RUCAIBox/WildBench)
  - **RepoBench**: [Code understanding and repository-level tasks](https://github.com/mlfoundations/RepoBench)
  - **MixEval**: [Comprehensive evaluation across domains](https://github.com/Thartvigsen/MixEval)
  - **IFEval**: [Instruction following capability evaluation](https://github.com/OpenBMB/IFEval)
  - **AlpacaEval**: [Instruction following evaluation](https://github.com/tatsu-lab/alpaca_eval)
  - **HumanEval**: [Code generation and problem solving](https://github.com/openai/human-eval)
  - **ZeroEval**: [Logical reasoning and problem solving](https://github.com/allenai/zero-eval)
  - **MBPP**: [Python programming benchmark](https://github.com/google-research/google-research/tree/master/mbpp)
  - **Arena-Hard-Auto** (Coming soon): [Automatic evaluation tool for instruction-tuned LLMs](https://github.com/lmarena/arena-hard-auto)
  - **SWE-Bench** (Coming soon): [Evaluating large language models on real world software issues](https://github.com/princeton-nlp/SWE-bench)
  - **SafetyBench** (Coming soon): [Evaluating the safety of LLMs](https://github.com/thu-coai/SafetyBench)
  - **Berkeley Function Calling Leaderboard** (Coming soon): [Evaluating ability of LLMs to use APIs](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)


### Basic Usage

Make sure your `OPENAI_API_KEY` is set in your environment before running evaluations.

```bash
python -m eval.eval \
    --model hf \
    --tasks HumanEval,mmlu \
    --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size 16 \
    --output_path logs
```

**Args**: 
- `--model`: Model type (example: hf, vllm)
- `--tasks`: Comma-separated list of benchmarks to run
- `--model_args`: Model path and parameters
- `--batch_size`: Batch size for inference
- `--output_path`: Directory to save evaluation results

Example running multiple benchmarks:
```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,WildBench,alpaca_eval \
    --model_args "pretrained=meta-llama/Llama-3-8B-Instruct" \
    --batch_size 16 \
    --output_path logs
```

We add several examples in `eval/examples` of sample scripts in different use cases for our evaluation framework. 

## 🔧 Advanced Usage

### Support for different models

Through LM-Eval-Harness, we support a wide array of different models, including HuggingFace models, VLLM, OpenAI, and more. For more information on how to use such models, we list all models supported at the [models page](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/models).

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

> **💡 Note**: While "auto" batch size is supported, we recommend manually tuning the batch size for optimal performance. The optimal batch size depends on the model size, GPU memory, and the specific benchmark. We used a maximum of 32 and a minimum of 4 (for RepoBench) evaluating Llama-3-8B-Instruct on 8xH100 GPUs.

### Customizing Evaluation

#### 🤖 Change Annotator Model

As part of our framework, we want to make swapping in different Language Model Judges for common benchmarks easy. Currently, we support two judge settings. The first is the default setting, where we use a benchmark's default judge. To activate this, you can either do nothing or pass in
```bash
--annotator_model auto
```
In addition to the default assignments, we support using gpt-4o-mini-2024-07-18 as a judge:

```bash
--annotator_model gpt-4o-mini-2024-07-18
```


### ⏱️ Runtime and Cost Analysis

Our framework makes running common benchmarks simple, fast, and versatile! We list the speeds and costs for each benchmark that we achieve with our framework for Llama-3-8B-Instruct on 8xH100 GPUs.

| Benchmark | Runtime (8xH100) | Batch Size | Total Tokens | API Cost | Notes |
|-----------|------------------|------------|--------------|-----------|--------|
| MTBench | 14:00 | 32 | ~196K | $0.05-6.40 | Varies by judge (GPT-4 vs GPT-4-mini) |
| WildBench | 38:00 | 32 | ~2.2M | $0.43 | Using GPT-4-mini judge |
| RepoBench | 46:00 | 4 | - | - | Lower batch size due to memory |
| MixEval | 13:00 | 32 | ~4-6M | $0.76-3.36 | Varies by judge model |
| AlpacaEval | 16:00 | 32 | ~936K | $0.14-9.40 | Varies by judge (GPT-4 vs GPT-4-mini) |
| HumanEval | 4:00 | 32 | - | - | No API costs |
| ZeroEval | 1:44:00 | 32 | - | - | Longest runtime |
| MBPP | 6:00 | 32 | - | - | No API costs |
| MMLU | 7:00 | 32 | - | - | No API costs |
| ARC | 4:00 | 32 | - | - | No API costs |
| Drop | 20:00 | 32 | - | - | No API costs |

**Notes:**
- Runtimes measured using 8x H100 GPUs with Llama-3 8B Instruct model
- Batch sizes optimized for memory and speed
- API costs vary based on judge model choice

**Cost-Saving Tips:**
- Use GPT-4-mini judge when possible for significant cost savings
- Adjust batch size based on available memory
- Consider using data-parallel evaluation for faster results

### 🔐 Special Access Requirements

#### ZeroEval Access
To run ZeroEval benchmarks, you need to:

1. Request access to the [ZebraLogicBench-private dataset](https://huggingface.co/datasets/allenai/ZebraLogicBench-private) on Hugging Face
2. Accept the terms and conditions
3. Log in to your Hugging Face account when running evaluations

## 🛠️ Implementing Custom Evaluations

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

### 🔍 Debug Mode

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

### 🚀 Performance Tips

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

### 🔧 Troubleshooting
Evalchemy has been tested on CUDA 12.4. If you run into issues like this: `undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12`, try updating your CUDA version:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

## 🏆 Leaderboard Integration

### 🗄️ Database Schema

We support automatically logging evaluation results to a unified PostgreSQL database. To enable logging to such a database, please use the "--use_database" flag (which defaults to False)
```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Llama-3-8B-Instruct' \
    --batch_size 2 \
    --output_path logs \
    --use_database
```
This requires the user set up a PostgreSQL database with the following comprehensive tables:

#### Models Table
```
- id: UUID primary key
- name: Model name
- base_model_id: Reference to parent model
- created_by: Creator of the model
- creation_location: Where model was created
- creation_time: When model was created
- training_start: Start time of training
- training_end: End time of training
- training_parameters: JSON of training configuration
- training_status: Current status of training
- dataset_id: Reference to training dataset
- is_external: Whether model is external
- weights_location: Where model weights are stored
- wandb_link: Link to Weights & Biases dashboard
- git_commit_hash: Model version in HuggingFace
- last_modified: Last modification timestamp
```

#### EvalResults Table
```
- id: UUID primary key
- model_id: Reference to evaluated model
- eval_setting_id: Reference to evaluation configuration
- score: Evaluation metric result
- dataset_id: Reference to evaluation dataset
- created_by: Who ran the evaluation
- creation_time: When evaluation was run
- creation_location: Where evaluation was run
- completions_location: Where outputs are stored
```

#### EvalSettings Table
```
- id: UUID primary key
- name: Setting name
- parameters: JSON of evaluation parameters
- eval_version_hash: Version hash of evaluation code
- display_order: Order in leaderboard display
```

#### Datasets Table
```
- id: UUID primary key
- name: Dataset name
- created_by: Creator of dataset
- creation_time: When dataset was created
- creation_location: Where dataset was created
- data_location: Storage location (S3/GCS/HuggingFace)
- generation_parameters: YAML pipeline configuration
- dataset_type: Type of dataset (SFT/RLHF)
- external_link: Original dataset source URL
- data_generation_hash: Fingerprint of dataset
- hf_fingerprint: HuggingFace fingerprint
```

### Database Configuration

#### PostgreSQL Setup
1. Install PostgreSQL on your system
2. Create a new database for Evalchemy
3. Create a user with appropriate permissions
4. Initialize the database schema using our models

#### Configure Database Connection
Set the following environment variables to enable database logging:

To enable using your own database, we recomend setting up a postgres-sql database with the following parameters. 
```bash
export DB_PASSWORD=<DB_PASSWORD>
export DB_HOST=<DB_HOST>
export DB_PORT=<DB_PORT>
export DB_NAME=<DB_NAME>
export DB_USER=<DB_USER>
```

### 📊 Submit Results to Leaderboard

```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Llama-3-8B-Instruct' \
    --batch_size 2 \
    --output_path logs \
    --use_database \
    --model_name "My Model Name" \
    --creation_location "Lab Name" \
    --created_by "Researcher Name"
```

View results on the [leaderboard](https://llm-leaderboard-319533213591.us-central1.run.app/).

### 🔄 Updating Database Results

You can update existing results using either:

1. Model ID: `--model_id <YOUR_MODEL_ID>`
2. Model Name: `--model-name <MODEL_NAME_IN_DB>`

Note: If both are provided, model_id takes precedence.
