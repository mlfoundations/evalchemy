import logging
import multiprocessing as mp

from bespokelabs.curator import LLM
from datasets import load_dataset
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

# This is just a demonstration of sharding.
# Really you want to run eval model weights with multiple GPUs and nodes
# Each node you use a different shard index
# Using curator with API model it is faster to just run a single curator instance, no shards

num_shards = 4  # Tested with 1, 2, 8, 16, 32
ds_name = "mlfoundations-dev/REASONING_evalchemy"
output_ds_name = f"{ds_name}_{num_shards}_shards_gpt-4o-mini"


class Answer(LLM):
    def prompt(self, row):
        return row["context"]

    def parse(self, row, response):
        row["model_outputs"] = response
        return row


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=30, max=600),
    reraise=True,
)
def push_to_hub_with_retry(dataset, ds_name, shard_index):
    try:
        dataset.push_to_hub(ds_name, config_name=f"shard_{shard_index}")
    except Exception as e:
        print(f"Shard {shard_index} push failed, will retry: {str(e)}")
        raise


def process_shard(shard_index):
    ds = load_dataset(ds_name, split="train")
    ds = ds.shard(num_shards=num_shards, index=shard_index)
    answer = Answer(model_name="gpt-4o-mini")
    ds = answer(ds)

    try:
        push_to_hub_with_retry(ds, output_ds_name, shard_index)
        print(f"Shard {shard_index} pushed to hub")
    except Exception as e:
        print(f"Failed to push shard {shard_index} after all retries: {str(e)}")


if __name__ == "__main__":
    with mp.Pool() as pool:
        pool.map(process_shard, range(num_shards))
    print(f"Viewable at https://huggingface.co/datasets/{output_ds_name}")
    ds = load_dataset(output_ds_name, split="train")
    print(ds)
