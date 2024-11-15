import os
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
from datasets import load_dataset


# These are multi-gb files that are really slow to load, even though they are already mixed
# files_to_download = [
#     'cot_zs_submix_data.json',
#     'flan2021_zsnoopt_submix_data.json',
#     'flan2021_zsopt_submix_data.json',
#     'niv2_zs_submix_data.json',
#     't0_zsnoopt_submix_data.json',
#     't0_zsopt_submix_data.json'
# ]
# download_path = snapshot_download(
#     repo_id="Open-Orca/FLAN",
#     allow_patterns=files_to_download,
#     local_dir=os.path.join(root, 'Open-Orca-FLAN'),
#     repo_type="dataset"
# )


# Instead let's download the parquets from each subdirectory individually
local_root = "dcft/slimorca/data"
repo_id = "Open-Orca/FLAN"
directories_to_download = [
    "niv2_zsopt_data",  # 2.5 GB
    "flan_zsopt_data",  # 11.7 GB
    "flan_zsnoopt_data",  # 16.3 GB
    "t0_zsopt_data",  # 17.5 GB
    "t0_zsnoopt_data",  # 10.8 GB
    "cot_zsopt_data",  # 20 MB
]

files = list_repo_files(repo_id, repo_type="dataset")
# print(files)

for directory in directories_to_download:
    download_path = snapshot_download(
        repo_id=repo_id,
        # allow_patterns=f'{directory}/*',
        allow_patterns=f"{directory}/part.0.parquet",
        local_dir=os.path.join(local_root, "Open-Orca-FLAN"),
        repo_type="dataset",
    )

# Loop through the directories and download each one
for directory in directories_to_download:
    # List files in the subdirectory on the Hugging Face Hub
    files = list_repo_files(repo_id, repo_type="dataset")
    subdirectory_files = [f for f in files if f.startswith(f"{directory}/")]

    print(f"Files in {directory}: {len(subdirectory_files)}")

    subset = os.path.join(download_path, directory)
    single_shard_ds = load_dataset(subset, data_files=f"part.0.parquet", split="train")
    print(f"Samples in shard: {len(single_shard_ds):,}")
    print(f"Estimated dataset size: {len(single_shard_ds) * len(subdirectory_files):,}")
