# 🧪 Evalchemy

> A unified and easy-to-use toolkit for evaluating post-trained language models

![alt text](https://github.com/mlfoundations/evalchemy/blob/main/image.png)

Evalchemy is developed by the [DataComp community](https://datacomp.ai) and [Bespoke Labs](https://bespokelabs.ai)  and builds on the [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness).


## 🎉 What's New 

#### [2025.02.24] New Reasoning Benchmarks

- AIME25 and Alice in Wonderland have been added to [available benchmarks](https://github.com/mlfoundations/evalchemy?tab=readme-ov-file#built-in-benchmarks).

#### [2025.01.30] API Model Support

- [API models via Curator](https://github.com/bespokelabsai/curator/): With `--model curator` you can now evaluate with even more API based models via [Curator](https://github.com/bespokelabsai/curator/), including all those supported by [LiteLLM](https://docs.litellm.ai/docs/providers) 

```
  python -m eval.eval \
        --model curator  \
        --tasks AIME24,MATH500,GPQADiamond \
        --model_name "gemini/gemini-2.0-flash-thinking-exp-01-21" \
        --apply_chat_template False \
        --model_args 'tokenized_requests=False' \
        --output_path logs
```

Here are other examples of `model_name`:
- `"claude-3-7-sonnet-latest-thinking"`
- `"deepseek-reasoner"`
- `"gemini/gemini-1.5-flash"`
- `"claude-3-7-sonnet-latest"`
- `"gpt-4o-mini-2024-07-18"`
- `"o1-preview-2024-09-12"`
- `"gpt-4o-2024-08-06"`

You can also change the `model_args` to fit your needs. For example, `"claude-3-7-sonnet-latest-thinking"` might need more tokens and more time for its thinking process and can be used in batch mode to speed up evaluation and reduce costs by setting `model_args` like this:

```
--model_args 'tokenized_requests=False,timeout=2000,max_length=64000,batch=True'
```

#### [2025.01.29] New Reasoning Benchmarks

- AIME24, AMC23, MATH500, LiveCodeBench, GPQADiamond, HumanEvalPlus, MBPPPlus, BigCodeBench, MultiPL-E, and CRUXEval have been added to our growing list of [available benchmarks](https://github.com/mlfoundations/evalchemy?tab=readme-ov-file#built-in-benchmarks). This is part of the effort in the [Open Thoughts](https://github.com/open-thoughts/open-thoughts) project. See the [our blog post](https://www.open-thoughts.ai/blog/measure) on using Evalchemy for measuring reasoning models. 

#### [2025.01.28] New Model Support
- [vLLM models](https://blog.vllm.ai/2023/06/20/vllm.html): High-performance inference and serving engine with PagedAttention technology
```bash
python -m eval.eval \
    --model vllm \
    --tasks alpaca_eval \
    --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size 16 \
    --output_path logs
```
- [OpenAI models](https://openai.com/): Full support for OpenAI's model lineup
```bash
python -m eval.eval \
    --model openai-chat-completions \
    --tasks alpaca_eval \
    --model_args "model=gpt-4o-mini-2024-07-18,num_concurrent=32" \
    --batch_size 16 \
    --output_path logs 
```

### Key Features

- **Unified Installation**: One-step setup for all benchmarks, eliminating dependency conflicts
- **Parallel Evaluation**:
  - Data-Parallel: Distribute evaluations across multiple GPUs for faster results
  - Model-Parallel: Handle large models that don't fit on a single GPU
- **Simplified Usage**: Run any benchmark with a consistent command-line interface
- **Results Management**: 
  - Local results tracking with standardized output format
  - Optional database integration for systematic tracking
  - Leaderboard submission capability (requires database setup)

## ⚡ Quick Start

### Installation

We suggest using conda ([installation instructions](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)). 

```bash
# Create and activate conda environment
conda create --name evalchemy python=3.10
conda activate evalchemy

# Clone the repo
git clone git@github.com:mlfoundations/evalchemy.git   
cd evalchemy

# Install dependencies
pip install -e .
pip install -e eval/chat_benchmarks/alpaca_eval

# Note: On some HPC systems you may need to modify pyproject.toml 
# to use absolute paths for the fschat dependency:
# Change: "fschat @ file:eval/chat_benchmarks/MTBench"
# To:     "fschat @ file:///absolute/path/to/evalchemy/eval/chat_benchmarks/MTBench"
# Or remove entirely and separately run
# pip install -e eval/chat_benchmarks/MTBench 

# Log into HuggingFace for datasets and models.
huggingface-cli login
```

## 📚 Available Tasks

### Built-in Benchmarks
- All tasks from [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Custom instruction-based tasks (found in [`eval/chat_benchmarks/`](eval/chat_benchmarks/)):
  - **MTBench**: [Multi-turn dialogue evaluation benchmark](https://github.com/mtbench101/mt-bench-101)
  - **WildBench**: [Real-world task evaluation](https://github.com/allenai/WildBench)
  - **RepoBench**: [Code understanding and repository-level tasks](https://github.com/Leolty/repobench)
  - **MixEval**: [Comprehensive evaluation across domains](https://github.com/Psycoy/MixEval)
  - **IFEval**: [Instruction following capability evaluation](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
  - **AlpacaEval**: [Instruction following evaluation](https://github.com/tatsu-lab/alpaca_eval)
  - **HumanEval**: [Code generation and problem solving](https://github.com/openai/human-eval)
  - **HumanEvalPlus**: [HumanEval with more test cases](https://github.com/evalplus/evalplus)
  - **ZeroEval**: [Logical reasoning and problem solving](https://github.com/WildEval/ZeroEval)
  - **MBPP**: [Python programming benchmark](https://github.com/google-research/google-research/tree/master/mbpp)
  - **MBPPPlus**: [MBPP with more test cases](https://github.com/evalplus/evalplus)
  - **BigCodeBench:** [Benchmarking Code Generation with Diverse Function Calls and Complex Instructions](https://arxiv.org/abs/2406.15877)

    > **🚨 Warning:** for BigCodeBench evaluation, we strongly recommend using a Docker container since the execution of LLM generated code on a machine can lead to destructive outcomes. More info is [here](eval/chat_benchmarks/BigCodeBench/README.md).
  - **MultiPL-E:** [Multi-Programming Language Evaluation of Large Language Models of Code](https://github.com/nuprl/MultiPL-E/)
  - **CRUXEval:** [Code Reasoning, Understanding, and Execution Evaluation](https://arxiv.org/abs/2401.03065)
  - **AIME24**: [Math Reasoning Dataset](https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024)
  - **AIME25**: [Math Reasoning Dataset](https://huggingface.co/datasets/TIGER-Lab/AIME25)
  - **AMC23**: [Math Reasoning Dataset](https://huggingface.co/datasets/AI-MO/aimo-validation-amc)
  - **MATH500**: [Math Reasoning Dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) split from [Let's Verify Step by Step](https://github.com/openai/prm800k/tree/main?tab=readme-ov-file#math-splits)
  - **LiveCodeBench**: [Benchmark of LLMs for code](https://livecodebench.github.io/)
  - **LiveBench**: [A benchmark for LLMs designed with test set contamination and objective evaluation in mind](https://livebench.ai/#/)
  - **GPQA Diamond**: [A Graduate-Level Google-Proof Q&A Benchmark](https://huggingface.co/datasets/Idavidrein/gpqa)
  - **Alice in Wonderland**: [Simple Tasks Showing Complete Reasoning Breakdown in LLMs](https://arxiv.org/abs/2406.02061)
  - **Arena-Hard-Auto** (Coming soon): [Automatic evaluation tool for instruction-tuned LLMs](https://github.com/lmarena/arena-hard-auto)
  - **SWE-Bench** (Coming soon): [Evaluating large language models on real-world software issues](https://github.com/princeton-nlp/SWE-bench)
  - **SafetyBench** (Coming soon): [Evaluating the safety of LLMs](https://github.com/thu-coai/SafetyBench)
  - **SciCode Bench** (Coming soon): [Evaluate language models in generating code for solving realistic scientific research problems](https://github.com/scicode-bench/SciCode)
  - **Berkeley Function Calling Leaderboard** (Coming soon): [Evaluating ability of LLMs to use APIs](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)
  

We have recorded reproduced results against published numbers for these benchmarks in [`reproduced_benchmarks.md`](reproduced_benchmarks.md).


### Basic Usage

Make sure your `OPENAI_API_KEY` is set in your environment before running evaluations, if an LLM judge is required. 

```bash
python -m eval.eval \
    --model hf \
    --tasks HumanEval,mmlu \
    --model_args "pretrained=mistralai/Mistral-7B-Instruct-v0.3" \
    --batch_size 2 \
    --output_path logs
```

The results will be written out in `output_path`. If you have `jq` [installed](https://jqlang.github.io/jq/download/), you can view the results easily after evaluation. Example: `jq '.results' logs/Qwen__Qwen2.5-7B-Instruct/results_2024-11-17T17-12-28.668908.json`

**Args**: 

- `--model`: Which model type or provider is evaluated (example: hf)
- `--tasks`: Comma-separated list of tasks to be evaluated.
- `--model_args`: Model path and parameters. Comma-separated list of parameters passed to the model constructor. Accepts a string of the format `"arg1=val1,arg2=val2,..."`. You can find the list supported arguments [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/365fcda9b85bbb6e0572d91976b8daf409164500/lm_eval/models/huggingface.py#L66).
- `--batch_size`: Batch size for inference
- `--output_path`: Directory to save evaluation results

Example running multiple benchmarks:
```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,WildBench,alpaca_eval \
    --model_args "pretrained=mistralai/Mistral-7B-Instruct-v0.3" \
    --batch_size 2 \
    --output_path logs
```

**Config shortcuts**: 

To be able to reuse commonly used settings without having to manually supply full arguments every time, we support reading eval configs from YAML files. These configs replace the `--batch_size`, `--tasks`, and `--annoator_model` arguments. Some example config files can be found in `./configs`. To use these configs, you can use the `--config` flag as shown below:

```bash
python -m eval.eval \
    --model hf \
    --model_args "pretrained=mistralai/Mistral-7B-Instruct-v0.3" \
    --output_path logs \
    --config configs/light_gpt4omini0718.yaml
```

We add several more command examples in [`eval/examples`](https://github.com/mlfoundations/Evalchemy/tree/main/eval/examples) to help you start using Evalchemy. 

## 🔧 Advanced Usage

### Support for different models

Through LM-Eval-Harness, we support all HuggingFace models and are currently adding support for all LM-Eval-Harness models, such as OpenAI and VLLM. For more information on such models, please check out the [models page](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/models).

To choose a model, simply set 'pretrained=<name of hf model>' where the model name can either be a HuggingFace model name or a path to a local model. 


### HPC Distributed Evaluation

For even faster evaluation, use full data parallelism and launch a vLLM process for each GPU. 

We have made also made this easy to do at scale across multiple nodes on HPC (High-Performance Computing) clusters:

```bash
python eval/distributed/launch.py --model_name <model_id> --tasks <task_list> --num_shards <n> --watchdog
```

Key features:
- Run evaluations in parallel across multiple compute nodes
- Dramatically reduce wall clock time for large benchmarks
- Offline mode support for environments without internet access on GPU nodes
- Automatic cluster detection and configuration
- Efficient result collection and scoring

Refer to the [distributed README](eval/distributed/README.md) for more details. 

NOTE: This is configured for specific HPC clusters, but can easily be adapted. Furthermore it can be adapted for a non-HPC setup using `CUDA_VISIBLE_DEVICES`