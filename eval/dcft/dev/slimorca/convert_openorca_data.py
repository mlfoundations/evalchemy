import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


import argparse

parser = argparse.ArgumentParser(description="Convert OpenOrca data")
parser.add_argument("--root", type=str, required=True, help="Root directory of the input data")
parser.add_argument("--save_root", type=str, required=True, help="Root directory to save the converted data")
args = parser.parse_args()

root = args.root
save_root = args.save_root

os.makedirs(save_root, exist_ok=True)

for subfolder in tqdm(os.listdir(root)):
    # for subfolder in tqdm(['flan_zsnoopt_data']):
    os.makedirs(os.path.join(save_root, subfolder), exist_ok=True)
    task_name_number = defaultdict(int)
    for file in tqdm(os.listdir(os.path.join(root, subfolder))):
        filename = os.path.join(root, subfolder, file)
        save_filename = os.path.join(save_root, subfolder, file.replace(".parquet", ".jsonl"))
        if not os.path.exists(save_filename):
            try:
                df = pd.read_parquet(filename, engine="pyarrow")
                instances = []
                for i in tqdm(range(len(df))):
                    row = df.iloc[i]
                    instance = {
                        "instruction": row["inputs"],
                        "output": row["targets"],
                        "metadata": {
                            "_template_idx": row["_template_idx"],
                            "_task_source": row["_task_source"],
                            "_task_name": row["_task_name"],
                            "_template_type": row["_template_type"],
                        },
                    }
                    task_name_number[row["_task_name"]] += 1
                    instances.append(instance)
                for instance in tqdm(instances):
                    with open(save_filename, "a") as f:
                        strg = json.dumps(instance, cls=NpEncoder)
                        f.write(strg)
                        f.write("\n")
            except:
                continue
    save_folder_task_number_filename = os.path.join(save_root, f"{subfolder}_task_name_number.json")
    if not os.path.exists(save_folder_task_number_filename):
        with open(save_folder_task_number_filename, "w") as f:
            json.dump(task_name_number, f, indent=4)
