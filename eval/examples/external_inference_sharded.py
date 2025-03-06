import logging
import multiprocessing as mp
import tempfile
from functools import partial

from bespokelabs.curator import LLM
from datasets import load_dataset
from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_exponential


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
def commit_with_retry(dataset, api, repo_id, shard_num, num_shards):
    """Push dataset to Hugging Face Hub with automatic retries."""
    try:
        # Create temporary file and save the dataset
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            dataset.to_parquet(tmp.name)
            # Format the filename for the shard
            shard_filename = f"train-{shard_num:05d}-of-{num_shards:05d}.parquet"
            # Upload the file
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=shard_filename,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Adding shard {shard_num}",
            )

        print(f"Successfully pushed shard {shard_num} to {repo_id} as {shard_filename}")
    except Exception as e:
        print(f"Push failed for shard {shard_num}, will retry: {str(e)}")
        raise


def process_shard(shard_index, num_shards, api, output_ds_name, ds):
    shard = ds.shard(num_shards=num_shards, index=shard_index)
    answer = Answer(model_name="gpt-4o-mini")
    shard = answer(shard)

    try:
        commit_with_retry(shard, api, output_ds_name, shard_index, num_shards)
        print(f"Shard {shard_index} pushed to hub")
    except Exception as e:
        print(f"Failed to push shard {shard_index} after all retries: {str(e)}")


if __name__ == "__main__":
    # This is just a demonstration of sharding.
    # Really you want to run eval model weights with multiple GPUs and nodes
    # Each node you use a different shard index
    # Using curator with API model it is faster to just run a single curator instance, no shards

    num_shards = 4  # Tested with 1, 2, 8, 16, 32
    ds_name = "mlfoundations-dev/REASONING_evalchemy"
    output_ds_name = f"{ds_name}_{num_shards}shards_gpt-4o-mini_diff"
    api = HfApi()
    ds = load_dataset(ds_name, split="train")

    # Check if repo exists before creating
    try:
        api.repo_info(repo_id=output_ds_name, repo_type="dataset")
        print(f"Repository {output_ds_name} already exists")
    except Exception:
        print(f"Creating new repository {output_ds_name}")
        api.create_repo(repo_id=output_ds_name, repo_type="dataset")

    with mp.Pool() as pool:
        pool.map(
            partial(process_shard, num_shards=num_shards, api=api, output_ds_name=output_ds_name, ds=ds),
            range(num_shards),
        )
    print(f"Viewable at https://huggingface.co/datasets/{output_ds_name}")
    ds = load_dataset(output_ds_name, split="train")
    print(ds)
