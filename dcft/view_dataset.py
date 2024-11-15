import argparse
import asyncio
import json
import os
from datetime import datetime
from functools import partial

import gcsfs
import psycopg

from datasets import concatenate_datasets, load_from_disk

OUTPUT_DIR = "gs://dcft-data-gcp/datasets-cache"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dcft/service_account_credentials.json"


async def load_shard(shard_path, custom_open):
    """Asynchronously load a single shard."""
    return load_from_disk(f"gs://{shard_path}", storage_options={"open": custom_open})


async def load_shards(dataset_id, num_shards=1):
    """
    Asynchronously load a specified number of shards for a given dataset ID from GCS
    and concatenate them into a single dataset.

    Args:
        dataset_id (str): The dataset ID to load shards for.
        num_shards (int): Number of shards to load.

    Returns:
        A concatenated dataset containing the loaded shards.
    """
    loop = asyncio.get_event_loop()
    fs = gcsfs.GCSFileSystem(project="bespokelabs", asynchronous=True, loop=loop)
    custom_open = partial(fs._open)

    dataset_dir = os.path.join(OUTPUT_DIR, dataset_id)
    shard_paths = await fs._ls(dataset_dir)
    shard_paths = [path for path in shard_paths if not path.endswith("info.json")]
    shard_paths = sorted(shard_paths)[:num_shards]

    print(f"Loading {len(shard_paths)} shards")

    tasks = [load_shard(path, custom_open) for path in shard_paths]
    shard_datasets = await asyncio.gather(*tasks)

    return concatenate_datasets(shard_datasets)


def get_datasets_by_name(db_connection_string, name):
    """
    Query the PostgreSQL database to get all datasets with the given name.

    Args:
        db_connection_string (str): The connection string for the PostgreSQL database.
        name (str): The name of the datasets to query.

    Returns:
        list: A list of tuples containing (id, generation_start, generation_end, generation_status, generation_parameters) for each matching dataset.
    """
    with psycopg.connect(db_connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, generation_start, generation_end, generation_status, generation_parameters FROM datasets WHERE name = %s ORDER BY creation_time DESC",
                (name,),
            )
            results = cur.fetchall()
            return [(str(row[0]), row[1], row[2], row[3], row[4]) for row in results]


def print_first_n_rows(file_path, n=5):
    """
    Pretty-print the first n rows from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.
        n (int): Number of rows to print.
    """
    print(f"\nFirst {n} rows of the dataset:")
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            print(f"\nRow {i+1}:")
            print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="View information about datasets with a given name or ID")
    parser.add_argument("identifier", help="The dataset ID or name to view")
    parser.add_argument(
        "--db-connection-string",
        default=f"postgresql://postgres:t%7DLQ7ZL%5D3%24x~I8ye@35.225.163.235:5432/postgres",
        help="PostgreSQL database connection string",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards to load (default: 1)")
    parser.add_argument("--output", default="output.jsonl", help="Output JSONL file name (default: output.jsonl)")

    args = parser.parse_args()

    if args.db_connection_string:
        # Try to get all datasets with the given name if a database connection string is provided
        datasets = get_datasets_by_name(args.db_connection_string, args.identifier)
        if datasets:
            print(f"Found {len(datasets)} datasets with name: {args.identifier}")
            print(
                "{:<5} | {:<36} | {:<19} | {:<19} | {:<15}".format(
                    "Index", "ID", "Generation Start", "Generation End", "Generation Status"
                )
            )
            print("-" * 105)
            for index, (id, start, end, status, _) in enumerate(datasets, start=1):
                start_str = start.strftime("%Y-%m-%d %H:%M:%S") if start else "N/A"
                end_str = end.strftime("%Y-%m-%d %H:%M:%S") if end else "N/A"
                status_str = status if status else "N/A"
                print("{:<5} | {:<36} | {:<19} | {:<19} | {:<15}".format(index, id, start_str, end_str, status_str))

            while True:
                choice = input("\nEnter the index of the dataset you want to select (or 'q' to quit): ")
                if choice.lower() == "q":
                    print("Exiting.")
                    return
                try:
                    index = int(choice)
                    if 1 <= index <= len(datasets):
                        selected_dataset = datasets[index - 1]
                        print(f"\nYou selected dataset:")
                        print(f"ID: {selected_dataset[0]}")
                        print(f"Generation Start: {selected_dataset[1]}")
                        print(f"Generation End: {selected_dataset[2]}")
                        print(f"Generation Status: {selected_dataset[3]}")
                        print("\nGeneration Parameters:")
                        print(json.dumps(selected_dataset[4], indent=2))

                        # Download the selected dataset
                        print(f"\nDownloading dataset shards...")
                        dataset = asyncio.run(load_shards(selected_dataset[0], args.num_shards))
                        print(f"Dataset has {len(dataset)} rows")
                        print(f"Saving dataset to {args.output}")
                        dataset.to_json(args.output, lines=True)
                        print(f"Saved successfully to {args.output}")

                        # Pretty-print the first 5 rows
                        print_first_n_rows(args.output, 5)

                        return
                    else:
                        print("Invalid index. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'q' to quit.")
        else:
            print(f"No datasets found with name: {args.identifier}.")
    else:
        print("Database connection string not provided. Unable to retrieve dataset information.")


if __name__ == "__main__":
    main()
