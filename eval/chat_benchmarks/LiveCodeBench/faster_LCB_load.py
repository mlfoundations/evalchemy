import os

from datasets import concatenate_datasets, load_dataset

from eval.chat_benchmarks.LiveCodeBench.livecodebench_utils import map_to_example, translate_private_test_cases

# USAGE:
# python eval/chat_benchmarks/LiveCodeBench/faster_LCB_load.py
cpu_count = os.cpu_count()
ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2", split="test", trust_remote_code=True)
# This results in pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays
# ds = ds.map(lambda example: {"private_test_cases": translate_private_test_cases(example["private_test_cases"])}, num_proc=cpu_count)
processed_shards = []
num_shards = 4
for i in range(num_shards):
    shard = ds.shard(num_shards=num_shards, index=i)
    shard = shard.map(
        lambda example: {"private_test_cases": translate_private_test_cases(example["private_test_cases"])},
        num_proc=cpu_count,
    )
    shard = shard.map(map_to_example, remove_columns=ds.column_names)
    processed_shards.append(shard)
ds = concatenate_datasets(processed_shards)
ds.push_to_hub("mlfoundations-dev/lcbv2_processed")
