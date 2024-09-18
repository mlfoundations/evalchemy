import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

from openai import AsyncOpenAI

from dcft.dataset.hf import get_dataclass_from_path
from dcft.dataset.reannotate import regenerate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BatchWatcher:
    def __init__(self, batch_objects_file, check_interval=60):
        self.client = AsyncOpenAI()
        with open(batch_objects_file, 'r') as f:
            self.batch_objects = json.load(f)
        self.batch_ids = [obj['id'] for obj in self.batch_objects]
        self.check_interval = check_interval

    async def check_batch_status(self, batch_id):
        batch = await self.client.batches.retrieve(batch_id)
        logging.info(f"Batch {batch_id} status: {batch.status} request_counts: {batch.request_counts}")
        return batch_id, batch.status

    async def watch(self):
        completed_batches = set()
        while len(completed_batches) < len(self.batch_ids):
            tasks = []
            for batch_id in self.batch_ids:
                if batch_id not in completed_batches:
                    tasks.append(self.check_batch_status(batch_id))
            
            results = await asyncio.gather(*tasks)
            
            for batch_id, status in results:
                if status in ["completed", "failed", "expired", "cancelled"]:
                    logging.info(f"Batch {batch_id} processing finished with status: {status}")
                    completed_batches.add(batch_id)

            if len(completed_batches) < len(self.batch_ids):
                logging.info(f"Sleeping for {self.check_interval} seconds...")
                await asyncio.sleep(self.check_interval)

        return [await self.client.batches.retrieve(batch_id) for batch_id in self.batch_ids]

    async def download_batch_result(self, batch_id):
        batch = await self.client.batches.retrieve(batch_id)
        if batch.status == "completed" and batch.output_file_id:
            file_content = await self.client.files.content(batch.output_file_id)
            return file_content.text.splitlines()
        return []

    async def download_results(self, output_path):
        tasks = [self.download_batch_result(batch_id) for batch_id in self.batch_ids]
        results = await asyncio.gather(*tasks)
        
        all_results = [item for sublist in results for item in sublist]  # Flatten the list of lists

        with open(output_path, "w") as f:
            for result in all_results:
                f.write(result + "\n")
        logging.info(f"All batch results downloaded and saved to: {output_path}")

    async def download_errors(self, error_path, batch_id):
        batch = await self.client.batches.retrieve(batch_id)
        if batch.error_file_id:
            file_content = await self.client.files.content(batch.error_file_id)
            with open(error_path, "wb") as f:
                f.write(file_content.content)
            logging.info(f"Batch errors downloaded and saved to: {error_path}")
        else:
            logging.info(f"No error file available for batch {batch_id}.")

async def watch_and_download(args):
    watcher = BatchWatcher(args.batch_objects_file)
    final_batches = await watcher.watch()
    logging.info(f"All batches completed")

    if all(batch.status == "completed" for batch in final_batches):
        await watcher.download_results(args.output_file)

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
            await watcher.download_errors(f"{args.error_file}.{batch.id}", batch.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch batch requests using their IDs and download results or errors.")
    parser.add_argument("--batch_objects_file", type=str, required=True, help="Path to the batch_objects.json file")
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

    asyncio.run(watch_and_download(args))
