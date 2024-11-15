from datasets import load_dataset

repo_id = "Open-Orca/FLAN"

# skipping zsnoopt data for now
directories_to_download = [
    "cot_zsopt_data",  # 20 MB
    "niv2_zsopt_data",  # 2.5 GB
    "flan_zsopt_data",  # 11.7 GB
    # 'flan_zsnoopt_data',  # 16.3 GB
    "t0_zsopt_data",  # 17.5 GB
    # 't0_zsnoopt_data',  # 10.8 GB
]

for directory in directories_to_download:
    full_ds = load_dataset(repo_id, data_dir=directory, split="train")
    print(f"Samples in {directory}: {len(full_ds):,}")
    del full_ds
