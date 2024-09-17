import argparse
import json
import os
import time
from typing import Any, Dict, List

from openai import OpenAI

from dcft.dataset.hf import get_dataclass_from_path
from dcft.dataset.reannotate import regenerate_dataset


class BatchWatcher:
    def __init__(self, batch_id, check_interval=60):
        self.client = OpenAI()
        self.batch_id = batch_id
        self.check_interval = check_interval

    def watch(self):
        while True:
            batch = self.client.batches.retrieve(self.batch_id)
            print(f"Current batch status: {batch.status}")

            if batch.status in ["completed", "failed", "expired", "cancelled"]:
                print(f"Batch processing finished with status: {batch.status}")
                return batch

            print(f"Sleeping for {self.check_interval} seconds...")
            time.sleep(self.check_interval)

    def download_results(self, output_path):
        batch = self.client.batches.retrieve(self.batch_id)
        if batch.status == "completed" and batch.output_file_id:
            file_content = self.client.files.content(batch.output_file_id)
            with open(output_path, "wb") as f:
                f.write(file_content.content)
            print(f"Batch results downloaded and saved to: {output_path}")
        else:
            print("Batch results are not available for download.")

    def download_errors(self, error_path):
        batch = self.client.batches.retrieve(self.batch_id)
        if batch.error_file_id:
            file_content = self.client.files.content(batch.error_file_id)
            with open(error_path, "wb") as f:
                f.write(file_content.content)
            print(f"Batch errors downloaded and saved to: {error_path}")
        else:
            print("No error file available for this batch.")


def watch(args):
    watcher = BatchWatcher(args.batch_id)
    final_batch = watcher.watch()
    print(f"Final batch details: {final_batch}")

    if final_batch.status == "completed":
        watcher.download_results(args.output_file)
        
        # Restore data object
        data = get_dataclass_from_path(args.dataset)
        data.user_prompts = data.user_prompts[:2]
        data.system_prompts = data.system_prompts[:2]
        n = len(data.user_prompts)

        # Process batch results
        outputs = {}
        with open(args.output_file, 'r') as f:
            for line in f:
                l = json.loads(line)
                outputs[int(l['custom_id'])] = (
                    l['response']['body']['choices'][0]['message']['content']
                )
        print(f"Number of outputs: {len(outputs)}")
        data.annotations = [outputs.get(i, {}) for i in range(n)]

        # Save updated data
        save_name = f"{args.dataset.replace('/', '_')}_{args.annotator}"
        save_dir_path = os.path.join(args.save_dir, save_name)
        os.makedirs(save_dir_path, exist_ok=True)
        save_path = os.path.join(save_dir_path, "reannotated.json")
        save_out = [
            {
                "system_prompt": data.system_prompts[idx],
                "user_prompt": data.user_prompts[idx],
                "annotation_original": data.annotations_original[idx],
                "annotation": data.annotations[idx],
            }
            for idx in range(len(data.annotations))
        ]
        with open(save_path, "w") as f:
            json.dump(save_out, f, indent=4)
        print(f"Updated data saved to {save_path}")

    if final_batch.error_file_id:
        watcher.download_errors(args.error_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a batch request using its ID and download results or errors.")
    parser.add_argument("--batch_id", type=str, help="The batch ID to watch")
    parser.add_argument(
        "--output_file",
        type=str,
        default="batch_results.jsonl",
        help="Path to save the batch results (default: batch_results.jsonl)",
    )
    parser.add_argument(
        "--error_file",
        type=str,
        default="batch_errors.jsonl",
        help="Path to save the batch errors (default: batch_errors.jsonl)",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to the original dataset")
    parser.add_argument("--save_dir", type=str, default="datasets/reannotated", help="Directory to save processed results")
    parser.add_argument("--annotator", type=str, default="gpt-4o-2024-08-06", help="Name of the annotator")

    args = parser.parse_args()

    watch(args)
