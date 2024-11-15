import os
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
from datasets import load_dataset


# Instead let's download the parquets from each subdirectory individually
local_root = "dcft/slimorca/data"
repo_id = "Open-Orca/FLAN"
directories_to_download = [
    "cot_zsopt_data"  # 20 MB
    "niv2_zsopt_data",  # 2.5 GB
    "flan_zsopt_data",  # 11.7 GB
    "flan_zsnoopt_data",  # 16.3 GB
    "t0_zsopt_data",  # 17.5 GB
    "t0_zsnoopt_data",  # 10.8 GB
]

for directory in directories_to_download:
    download_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{directory}/*",
        local_dir=os.path.join(local_root, "Open-Orca-FLAN"),
        repo_type="dataset",
    )

# There is duplicate when you load the dataset and generate samples... saved with arrow format to .cache/huggingface/datasets/...
for directory in directories_to_download:
    subset = os.path.join(download_path, directory)
    full_ds = load_dataset(subset, split="train")
    print(f"Samples in {directory}: {len(full_ds):,}")
    del full_ds
