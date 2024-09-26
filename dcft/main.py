from typing import Dict, Any, Optional
from dcft.data_strategies.__init__ import SyntheticDataManager
import argparse


def main(
    list_frameworks: bool = False,
    framework: Optional[str] = None,
    hf_account: Optional[str] = None,
    cache_dir: Optional[str] = None,
    fs_type: Optional[str] = None,
    overwrite_cache: Optional[str] = None,
) -> None:
    manager = SyntheticDataManager(
        hf_account=hf_account, cache_dir=cache_dir, fs_type=fs_type, overwrite_cache=overwrite_cache
    )

    if list_frameworks:
        print("Available frameworks:")
        for framework in manager.list_frameworks():
            print(f"  - {framework}")
    elif framework:
        manager.run_framework(framework)
    else:
        print("No action specified. Use --list to see available frameworks or --run to run a specific framework.")


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Framework Manager")
    parser.add_argument("--list", action="store_true", help="List all available frameworks")
    parser.add_argument("--framework", type=str, metavar="FRAMEWORK", help="Run a specific framework")
    parser.add_argument("--hf-account", type=str, help="HuggingFace account to upload dataset to")
    parser.add_argument("--cache-dir", type=str, help="Directory for caching")
    parser.add_argument("--overwrite-cache", action="store_true", help="Do we overwrite the cache with new values?")
    parser.add_argument("--fs-type", type=str, default="file", help="Do we overwrite the cache with new values?")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        list_frameworks=args.list,
        framework=args.framework,
        hf_account=args.hf_account,
        cache_dir=args.cache_dir,
        fs_type=args.fs_type,
        overwrite_cache=args.overwrite_cache,
    )
