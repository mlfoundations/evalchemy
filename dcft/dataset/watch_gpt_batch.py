import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List

from openai import OpenAI

from dcft.dataset.hf import get_dataclass_from_path
from dcft.dataset.reannotate import regenerate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BatchWatcher:
    def __init__(self, batch_ids, check_interval=60):
        self.client = OpenAI()
        self.batch_ids = batch_ids
        self.check_interval = check_interval

    def watch(self):
        completed_batches = []
        while len(completed_batches) < len(self.batch_ids):
            for batch_id in self.batch_ids:
                if batch_id in completed_batches:
                    continue
                batch = self.client.batches.retrieve(batch_id)
                logging.info(f"Batch {batch_id} status: {batch.status} " f"request_counts: {batch.request_counts}")

                if batch.status in ["completed", "failed", "expired", "cancelled"]:
                    logging.info(f"Batch {batch_id} processing finished with status: {batch.status}")
                    completed_batches.append(batch_id)

            if len(completed_batches) < len(self.batch_ids):
                logging.info(f"Sleeping for {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        return [self.client.batches.retrieve(batch_id) for batch_id in self.batch_ids]

    def download_results(self, output_path):
        all_results = []
        for batch_id in self.batch_ids:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed" and batch.output_file_id:
                file_content = self.client.files.content(batch.output_file_id)
                all_results.extend(file_content.content.decode().splitlines())

        with open(output_path, "w") as f:
            for result in all_results:
                f.write(result + "\n")
        logging.info(f"All batch results downloaded and saved to: {output_path}")

    def download_errors(self, error_path):
        batch = self.client.batches.retrieve(self.batch_id)
        if batch.error_file_id:
            file_content = self.client.files.content(batch.error_file_id)
            with open(error_path, "wb") as f:
                f.write(file_content.content)
            logging.info(f"Batch errors downloaded and saved to: {error_path}")
        else:
            logging.info("No error file available for this batch.")


def watch(args):
    watcher = BatchWatcher(args.batch_ids)
    final_batches = watcher.watch()
    logging.info(f"All batches completed")

    if all(batch.status == "completed" for batch in final_batches):
        watcher.download_results(args.output_file)

        # Restore data object
        data = get_dataclass_from_path(args.dataset)
        n = len(data.user_prompts)

        # Process batch results
        outputs = {}
        with open(args.output_file, "r") as f:
            for line in f:
                l = json.loads(line)
                outputs[int(l["custom_id"])] = l["response"]["body"]["choices"][0]["message"]["content"]
        logging.info(f"Number of outputs: {len(outputs)}")
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
        logging.info(f"Updated data saved to {save_path}")

    for batch in final_batches:
        if batch.error_file_id:
            watcher.download_errors(f"{args.error_file}.{batch.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch batch requests using their IDs and download results or errors.")
    parser.add_argument("--batch_ids", type=str, help="Comma-separated list of batch IDs to watch")
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
    parser.add_argument(
        "--save_dir", type=str, default="datasets/reannotated", help="Directory to save processed results"
    )
    parser.add_argument("--annotator", type=str, default="gpt-4o-2024-08-06", help="Name of the annotator")

    args = parser.parse_args()
    args.batch_ids = args.batch_ids.split(",")

    watch(args)
