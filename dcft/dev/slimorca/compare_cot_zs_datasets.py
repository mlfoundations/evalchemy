import os
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset


root = "dcft/slimorca/data"


download_path = snapshot_download(
    repo_id="Open-Orca/FLAN",
    allow_patterns="cot_zsopt_data/*",
    local_dir=os.path.join(root, "Open-Orca-FLAN"),
    repo_type="dataset",
)
datafiles = [
    os.path.join(download_path, "cot_zsopt_data", file)
    for file in os.listdir(os.path.join(download_path, "cot_zsopt_data"))
]
print(datafiles)
dataset = load_dataset(download_path, data_files=datafiles)["train"]
print(dataset)
print(dataset[0])

download_path = snapshot_download(
    repo_id="Open-Orca/FLAN",
    allow_patterns="cot_zs_submix_data.json",
    local_dir=os.path.join(root, "Open-Orca-FLAN"),
    repo_type="dataset",
)
datafiles = os.path.join(download_path, "cot_zs_submix_data.json")
print(datafiles)
dataset = load_dataset(download_path, data_files=datafiles)["train"]
print(dataset)
print(dataset[0])

download_path = snapshot_download(
    repo_id="SirNeural/flan_v2",
    allow_patterns="cot_zs*",
    local_dir=os.path.join(root, "SirNeural-flan_v2"),
    repo_type="dataset",
)
print(download_path)
dataset = load_dataset(download_path)["train"]
print(dataset)

# Load the dataset from the downloaded path
datafiles = [os.path.join(download_path, "cot_zs_noopt_train.jsonl.gz")]
print(datafiles)
noopt_dataset = load_dataset(download_path, data_files=datafiles)["train"]
print(noopt_dataset)
print(noopt_dataset[0])

datafiles = [os.path.join(download_path, "cot_zs_opt_train.jsonl.gz")]
print(datafiles)
opt_dataset = load_dataset(download_path, data_files=datafiles)["train"]
print(opt_dataset)
print(opt_dataset[0])

# SirNerual 150k is just the opt dataset twice. The original flan doesn't have noopt option for COT.
noopt_set = set((item["inputs"] + item["targets"] + item["task"]) for item in noopt_dataset)
opt_set = set((item["inputs"] + item["targets"] + item["task"]) for item in opt_dataset)

overlap = noopt_set & opt_set
unique_to_noopt = noopt_set - opt_set
unique_to_opt = opt_set - noopt_set

print(f"Number of overlapping items: {len(overlap)}")
print(f"Number of items unique to noopt dataset: {len(unique_to_noopt)}")
print(f"Number of items unique to opt dataset: {len(unique_to_opt)}")
