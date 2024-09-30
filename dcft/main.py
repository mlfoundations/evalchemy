import argparse
from typing import Any, Dict, Optional

from dcft.data_strategies.__init__ import SyntheticDataManager


def main(list_frameworks: bool = False, framework: Optional[str] = None, hf_account: Optional[str] = None) -> None:
    manager = SyntheticDataManager()

    if list_frameworks:
        print("Available frameworks:")
        for framework in manager.list_frameworks():
            print(f"  - {framework}")
    elif framework:
        manager.run_framework(framework, hf_account)
    else:
        print("No action specified. Use --list to see available frameworks or --run to run a specific framework.")


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Framework Manager")
    parser.add_argument("--list", action="store_true", help="List all available frameworks")
    parser.add_argument("--framework", type=str, metavar="FRAMEWORK", help="Run a specific framework")
    parser.add_argument("--hf-account", type=str, help="HuggingFace account to upload dataset to")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(list_frameworks=args.list, framework=args.framework, hf_account=args.hf_account)
