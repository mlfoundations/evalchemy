import multiprocessing as mp

from bespokelabs.curator import LLM
from datasets import load_dataset

# This is just a demonstration of sharding.
# Really you want to run eval model weights with multiple GPUs and nodes
# Each node you use a different shard index
# Using curator with API model it is faster to just run a single curator instance, no shards

num_shards = 8  # Tested with 1, 2, 8, 64
ds_name = "mlfoundations-dev/REASONING_evalchemy"
output_ds_name = f"{ds_name}_{num_shards}_sharded_gpt-4o-mini"


class Answer(LLM):
    def prompt(self, row):
        return row["context"]

    def parse(self, row, response):
        row["model_outputs"] = response
        return row


def process_shard(shard_index):
    ds = load_dataset(ds_name, split="train")
    ds = ds.shard(num_shards=num_shards, index=shard_index)
    answer = Answer(model_name="gpt-4o-mini")
    ds = answer(ds)

    ds.push_to_hub(output_ds_name, config_name=f"shard_{shard_index}")
    # note there could be issues with processes uploading at the same time
    # and also rate limits
    print(f"Viewable at https://huggingface.co/datasets/{output_ds_name}")


if __name__ == "__main__":
    with mp.Pool() as pool:
        pool.map(process_shard, range(num_shards))
    ds = load_dataset(output_ds_name, split="train")
    print(ds)
