# from Marianna, not specific to evalchemy, but uses curator and ray to anntoate.
import argparse
import itertools
import logging
import os
import time

import ray
import torch
from bespokelabs import curator
from datasets import load_dataset

logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)


def convert_row(row: dict) -> dict:
    conversation = row["conversations"]
    instruction = next((item["value"] for item in conversation if item["from"] == "human"), None)
    response = next((item["value"] for item in conversation if item["from"] == "gpt"), None)
    return {"instruction": instruction, "original_response": response}


def prompt_func(row):
    return row["instruction"]


def parse_func(row, response):
    instruction = row["instruction"]
    return {"instruction": instruction, "new_response": response}


@ray.remote
def generate_dataset(dataset_shard_list, prompter, output_path):
    distill_prompter = curator.LLM(
        prompt_func=prompt_func,
        parse_func=parse_func,
        model_name=model_name,
        batch=True,
        batch_size=256,
        backend="vllm",
        tensor_parallel_size=4,
        max_tokens=256,
        max_model_length=1024,
    )

    for dataset_shard_id in dataset_shard_list:
        dataset_shard = load_dataset("parquet", data_files=dataset_shard_id + ".parquet", split="train")
        distilled_dataset = distill_prompter(dataset_shard)
        torch.cuda.empty_cache()
        save_id = dataset_shard_id.split("/")[-1].split(".")[0]
        save_path = os.path.join(output_path, f"{dataset_shard_id}")
        distilled_dataset.to_parquet(f"{save_path}.parquet")
    return save_path + ".parquet"


def split_dataset(dataset, num_shards, shards_dir):
    shard_paths = []
    for i in range(num_shards):
        shard_path = f"{shards_dir}/shard_{i}"
        shard_paths.append(shard_path)
        dataset.shard(num_shards, index=i).to_parquet(f"{shard_path}.parquet")
    return shard_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=1000)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    model_name = args.model_name
    output_path = args.output_path

    num_nodes = args.num_nodes
    shard_size = args.shard_size

    if num_nodes > 1:
        ray.init(address="auto")

    dataset = load_dataset(dataset_path, split="train")
    dataset = dataset.take(num_nodes * shard_size)  # take the first num_nodes shards
    dataset = dataset.map(convert_row)
    dataset = dataset.select_columns(["instruction", "original_response"])

    num_shards = len(dataset) // shard_size
    num_shards_per_node = num_shards // num_nodes

    shards_dir = os.path.join(args.output_path, "__shards")
    os.makedirs(shards_dir, exist_ok=True)

    shard_paths = split_dataset(dataset, num_shards, shards_dir)

    shard_indices = list(
        itertools.chain(
            *[list(range(i * num_shards_per_node, (i + 1) * num_shards_per_node)) for i in range(num_nodes)]
        )
    )
    ret = []
    for i in range(num_nodes):
        shard_list = shard_indices[i * num_shards_per_node : (i + 1) * num_shards_per_node]
        shard_path_list = [shard_paths[shard_id] for shard_id in shard_list]
        ret.append(generate_dataset.options(num_gpus=1, num_cpus=8).remote(shard_path_list, prompt_func, output_path))

    shard_paths = ray.get(ret)
    ray.shutdown()
