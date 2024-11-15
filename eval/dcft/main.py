import argparse
import os
from typing import Any, Dict, Optional
import logging

from dcft.data_strategies.synthetic_data_manager import SyntheticDataManager

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dcft/service_account_credentials.json"


def main(
    list_frameworks: bool = False,
    framework: Optional[str] = None,
    hf_account: Optional[str] = None,
    remote: bool = False,
    no_return: bool = False,
    hf_private: Optional[bool] = True,
    fs_type: Optional[str] = None,
    ray_address: Optional[str] = None,
    max_pending_waitables: int = 100,
    output_dir: Optional[str] = None,
    db_connection_string: Optional[str] = None,
    enable_cache: bool = False,
    remote_caching: bool = False,
    log_level: str = "INFO",
    dev: bool = False,
    resume_from_partial: bool = False,
) -> None:
    if remote_caching or remote:
        output_dir = f"gs://dcft-data-gcp/datasets-cache"
        enable_cache = True
        fs_type = "gcs"

    if dev:
        enable_cache = True
        output_dir = "./datasets"
        db_connection_string = ""
        fs_type = "file"
        logging.warning("This is in developer mode, so this dataset will not be registered in the database.")

    manager = SyntheticDataManager(
        hf_account=hf_account,
        hf_private=hf_private,
        fs_type=fs_type,
        ray_address=ray_address,
        no_return=no_return,
        max_pending_waitables=max_pending_waitables,
        output_dir=output_dir,
        db_connection_string=db_connection_string,
        enable_cache=enable_cache,
        log_level=log_level,
        resume_from_partial=resume_from_partial,
    )

    if list_frameworks:
        print("Available frameworks:")
        for framework in manager.list_frameworks():
            print(f"  - {framework}")
    elif framework:
        if remote:
            manager.run_remote(framework)
        else:
            manager.run_framework(framework)
    else:
        print("No action specified. Use --list to see available frameworks or --run to run a specific framework.")


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Framework Manager")
    parser.add_argument("--list", action="store_true", help="List all available frameworks")
    parser.add_argument("--framework", type=str, metavar="FRAMEWORK", help="Run a specific framework")
    parser.add_argument("--hf-account", type=str, help="HuggingFace account to upload dataset to")
    parser.add_argument("--hf-private", action="store_true", help="Upload dataset to HuggingFace private repo")
    parser.add_argument("--fs-type", type=str, default="file", help="Filesystem type to use")
    parser.add_argument("--remote", action="store_true", help="Run the data generation process on a remote Ray cluster")
    parser.add_argument("--no-return", action="store_true", help="Whether to not return data to the local machine")
    parser.add_argument("--max-pending-waitables", type=int, default=100, help="Maximum number of pending waitables")
    parser.add_argument(
        "--ray-address",
        type=str,
        default="http://34.71.168.41:8265/",
        help="Address of the Ray client",
    )
    parser.add_argument("--output-dir", default="./datasets", type=str, help="Directory to output the dataset to")
    parser.add_argument(
        "--db-connection-string",
        type=str,
        default=f"postgresql://postgres:t%7DLQ7ZL%5D3%24x~I8ye@35.225.163.235:5432/postgres",
        help="Connection string for the PostgreSQL database",
    )
    parser.add_argument("--enable-cache", action="store_true", help="Enable caching")
    parser.add_argument("--dev", action="store_true", help="Enable developer settings")

    parser.add_argument("--remote-caching", action="store_true", help="Enable remote caching")
    parser.add_argument(
        "--resume-from-partial",
        action="store_true",
        help="Enables resume from partially completed operator's cache, even if said operator did not fully finish previously",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        list_frameworks=args.list,
        framework=args.framework,
        hf_account=args.hf_account,
        hf_private=args.hf_private,
        fs_type=args.fs_type,
        remote=args.remote,
        no_return=args.no_return,
        ray_address=args.ray_address,
        max_pending_waitables=args.max_pending_waitables,
        output_dir=args.output_dir,
        db_connection_string=args.db_connection_string,
        enable_cache=args.enable_cache,
        remote_caching=args.remote_caching,
        log_level=args.log_level,
        dev=args.dev,
        resume_from_partial=args.resume_from_partial,
    )
