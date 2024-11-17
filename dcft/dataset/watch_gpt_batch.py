import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

from openai import AsyncOpenAI

from dcft.dataset.hf import get_dataclass_from_path
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BatchWatcher:
    def __init__(self, batch_objects_file: str, check_interval: int = 60) -> None:
        """Initialize BatchWatcher with batch objects file and check interval.

        Args:
            batch_objects_file (str): Path to the batch objects JSON file.
            check_interval (int): Time interval (in seconds) to check batch status.
        """
        self.client = AsyncOpenAI()
        with open(batch_objects_file, "r") as f:
            self.batch_objects = json.load(f)
        self.batch_ids = [obj["id"] for obj in self.batch_objects]
        self.batches = []
        self.check_interval = check_interval

    async def check_batch_status(self, batch_id: str) -> tuple[str, str]:
        """Check the status of a batch by its ID.

        Args:
            batch_id (str): The ID of the batch to check.

        Returns:
            tuple[str, str]: The batch ID and its status.
        """
        batch = await self.client.batches.retrieve(batch_id)
        logging.info(
            f"Batch {batch_id} status: {batch.status} requests: {batch.request_counts.completed}/{batch.request_counts.failed}/{batch.request_counts.total} completed/failed/total"
        )
        return batch_id, batch

    async def watch(self) -> None:
        """Monitor the status of batches until all are completed (includes successfully, failed, expired or cancelled)."""
        completed_batches = {}
        while len(completed_batches) < len(self.batch_ids):
            status_tasks = []
            for batch_id in self.batch_ids:
                if batch_id not in completed_batches:
                    status_tasks.append(self.check_batch_status(batch_id))

            batches = await asyncio.gather(*status_tasks)
            for batch_id, batch in batches:
                if batch.status in ["completed", "failed", "expired", "cancelled"]:
                    logging.info(f"Batch {batch_id} processing finished with status: {batch.status}")
                    completed_batches[batch_id] = batch

            if len(completed_batches) < len(self.batch_ids):
                logging.info(
                    f"Remaining batches processing {len(self.batch_ids) - len(completed_batches)}/{len(self.batch_ids)}"
                )
                logging.info(f"Sleeping for {self.check_interval} seconds...")
                await asyncio.sleep(self.check_interval)

        self.batches = completed_batches.values()

    async def download_batch_result(self, batch) -> list[str]:
        """Download the result of a completed batch.

        Args:
            batch: The batch object to download results from.

        Returns:
            list[str]: Lines of the downloaded result.
        """
        if batch.status == "completed" and batch.output_file_id:
            file_content = await self.client.files.content(batch.output_file_id)
            return file_content.text.splitlines()
        return []

    async def download_results(self, output_path: str) -> None:
        """Download results of all batches and save to a specified path.

        Args:
            output_path (str): Path to save the downloaded results.
        """
        tasks = [self.download_batch_result(batch) for batch in self.batches]
        results = await asyncio.gather(*tasks)

        all_results = [item for sublist in results for item in sublist]  # Flatten the list of lists

        with open(output_path, "w") as f:
            for result in all_results:
                f.write(result + "\n")
        logging.info(f"All batch results downloaded and saved to: {output_path}")

    async def download_errors(self, error_path: str, batch_id: str) -> None:
        """Download error file for a specific batch if available.

        Args:
            error_path (str): Path to save the error file.
            batch_id (str): The ID of the batch to download errors from.
        """
        batch = await self.client.batches.retrieve(batch_id)
        if batch.error_file_id:
            file_content = await self.client.files.content(batch.error_file_id)
            with open(error_path, "wb") as f:
                f.write(file_content.content)
            logging.info(f"Batch errors downloaded and saved to: {error_path}")
        else:
            logging.info(f"No error file available for batch {batch_id}.")

    async def plot_completion_data(self, output_dir: str) -> None:
        """Save plots visualizing completion times for the batches.

        Args:
            output_dir (str): Directory to save the plots.
        """
        completion_times = []
        completion_dates = []

        for batch_id in self.batch_ids:
            batch = await self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                duration = (batch.completed_at - batch.created_at) / 60  # Convert to minutes
                completion_times.append(duration)
                completion_dates.append(batch.completed_at)

        # Create a DataFrame for plotting
        df = pd.DataFrame(
            {
                "Completion Time (min)": completion_times,  # Update label to minutes
                "Completion Date": pd.to_datetime(completion_dates, unit="s"),
            }
        )

        # Histogram of completion durations
        plt.figure(figsize=(12, 6))
        plt.hist(df["Completion Time (min)"], bins=20, color="blue", alpha=0.7)
        plt.title("Histogram of Completion Durations")
        plt.xlabel("Duration (minutes)")  # Update label to minutes
        plt.ylabel("Frequency")
        plt.grid(axis="y")
        plt.savefig(os.path.join(output_dir, "completion_durations_histogram.png"))  # Save the histogram
        plt.close()  # Close the plot

        # Cumulative plot of completed jobs over time
        df.sort_values("Completion Date", inplace=True)
        df["Cumulative Completed"] = range(1, len(df) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(df["Completion Date"], df["Cumulative Completed"], marker="o", color="green")
        plt.title("Cumulative Completed Jobs Over Time")
        plt.xlabel("Completion Date")
        plt.ylabel("Cumulative Completed Jobs")
        plt.grid()
        plt.savefig(os.path.join(output_dir, "cumulative_completed_jobs.png"))  # Save the cumulative plot
        plt.close()  # Close the plot


async def watch_and_download(batch_objects_file: str, batch_output_file: str, dataset: str, save_path: str) -> None:
    """Watch batches and download results when all are completed.

    Args:
        batch_objects_file (str): Path to the batch_objects.json file.
        batch_output_file (str): Path to save the batch results.
        save_path (str): Path to save the reannotated data.
    """
    watcher = BatchWatcher(batch_objects_file)
    await watcher.watch()
    logging.info(f"All batches completed")

    await watcher.download_results(batch_output_file)
    logging.info(f"Downloading results")

    # Restore data object
    data = get_dataclass_from_path(dataset)
    n = len(data.user_prompts)

    # Process batch results
    outputs = {}
    with open(batch_output_file, "r") as f:
        for line in f:
            l = json.loads(line)
            outputs[int(l["custom_id"])] = l["response"]["body"]["choices"][0]["message"]["content"]
    logging.info(f"Number of outputs: {len(outputs)}")
    data.annotations = [outputs.get(i, {}) for i in range(n)]

    # Save updated data
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

    for batch in watcher.batches:
        if batch.error_file_id:
            await watcher.download_errors(f"{args.error_file}.{batch.id}", batch.id)


async def watch_and_plot(batch_objects_file: str, output_dir: str) -> None:
    """Watch batches and plot completion data when all are completed.

    Args:
        batch_objects_file (str): Path to the batch_objects.json file.
        output_dir (str): Directory to save the plots.
    """
    watcher = BatchWatcher(batch_objects_file)
    await watcher.plot_completion_data(output_dir)  # Pass output_dir directly


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
    parser.add_argument("--graph", action="store_true", help="Plot completion data")

    args = parser.parse_args()

    if args.graph:
        asyncio.run(watch_and_plot(args.batch_objects_file, args.save_dir))
    else:
        save_name = f"{args.dataset.replace('/', '_')}_{args.annotator}"
        save_dir_path = os.path.join(args.save_dir, save_name)
        os.makedirs(save_dir_path, exist_ok=True)
        save_path = os.path.join(save_dir_path, "reannotated.json")
        asyncio.run(watch_and_download(args.batch_objects_file, args.output_file, args.dataset, save_path))
