# DCFT Training Guide

Our main training framework is a modified version of [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). LlamaFacotry is added as a submodule with minimal changes. 

Note: In general, all commands below are meant to be called from the top-level directory. (i.e., no need to `cd` inside any directories.)

## Requirements
LLamaFactory runs on Python 3.10. Install requirements.txt with the command below. This is taken from LlamaFactory's requirements with a few additional ones added (e.g. liger-kernel)
```bash
pip install -r dcft/train/requirements.txt
```
There's nothing really special or specific in the requirements. These should be relatively flexible.

## Quickstart
We have provided a sample yaml file to get started with. All training configs are contained within such yaml files, and each run will be associated with its own yaml file. 
<br>
**Note:** You may need to change `output_dir` in the yaml files. This is the local path where the model will be saved. The default is `experiments/train/checkpoints/(run_name)` 
```bash
torchrun --nnodes 1 --nproc_per_node 8 dcft/train/llamafactory/src/train.py dcft/train/configs/sample.yaml
```

**Multi-node**

Our multi-node setup uses torchrun. It requires the user to indicate `master_addr` and `master_port` as shown below.
```bash
torchrun --nnodes 1  --node_rank 0 --nproc_per_node 8 --master_addr {master_addr} --master_port {master_port} dcft/train/llamafactory/src/train.py dcft/train/configs/sample.yaml
```

## Key parameters in the YAML
- `model_name_or_path`: HF base model (e.g. mistralai/Mistral-7B-v0.1)
- `dataset_dir`: DATABASE or ONLINE
- `dataset`: If `dataset_dir` is DATABASE: this is the uuid of the dataset. If `dataset_dir` is ONLINE, this is the HF path of the dataset (e.g. mlfoundations-dev/oh-dcft-v1-no-curation-sharegpt-format)
- `template`: use `mistral` for Mistral, `llama3` for Llama3
- `formatting`: options are `alpaca` or `ShareGPT`
- `output_dir`: local path to store the checkpoints. If you don't want to save local checkpoints, set `save_strategy: "no"`
- `push_to_db`: if True, push to DB.
- `push_to_hub`: if True, push to HuggingFace.
- `export_hub_model_id`: HF repo name the model will be pushed to (e.g. mlfoundations-dev/mistral_alpaca_sft_sample)

## Presets
 
You can set 'include_hp' to reference the path of a Hyperparameter preset to improve reproducibility. Example would be 

```yaml
    include_hp: dcft/train/hp_settings/hritik.yaml
```


## Benchmarking
To make sure your runs are running smoothly, we have provided a few benchmarks in [benchmarks.md](dcft/train/benchmarks.md). In general, on an 80GB A100, running 1 epoch of LlamaFactory with `sample.yaml` should take around 3.5 minutes.
