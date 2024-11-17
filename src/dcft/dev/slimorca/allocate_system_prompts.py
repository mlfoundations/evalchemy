import os
import json
import random
import argparse
from tqdm import tqdm
from constants import *
from collections import defaultdict

parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument("--root", type=str, required=True, help="Root directory path")
parser.add_argument("--save_root", type=str, required=True, help="Save root directory path")
args = parser.parse_args()

root = args.root
save_root = args.save_root

files = os.listdir(root)

count = defaultdict(int)

for filepath in tqdm(files):
    if ".cache" in filepath:
        continue
    path = os.path.join(root, filepath)
    with open(path, "r") as f:
        data = list(f)

    instances = []
    if "cot_zsopt_data" in filepath:
        indices = MAPPING_ORCA_TASK_INDEX["cot"]
        system_prompts = [ORCA_SYSTEM_PROMPTS[index] for index in indices]
        for instance in tqdm(data):
            instance = eval(instance)
            chosen_sys_instruction = random.choice(system_prompts)
            instance["system_instruction"] = chosen_sys_instruction
            instances.append(instance)
    elif "niv2_zsopt_data" in filepath:
        indices = MAPPING_ORCA_TASK_INDEX["niv2"]
        system_prompts = [ORCA_SYSTEM_PROMPTS[index] for index in indices]
        for instance in tqdm(data):
            instance = eval(instance)
            chosen_sys_instruction = random.choice(system_prompts)
            instance["system_instruction"] = chosen_sys_instruction
            instances.append(instance)
    elif "t0_zsopt_data" in filepath:
        indices = MAPPING_ORCA_TASK_INDEX["t0"]
        system_prompts = [ORCA_SYSTEM_PROMPTS[index] for index in indices]
        for instance in tqdm(data):
            instance = eval(instance)
            chosen_sys_instruction = random.choice(system_prompts)
            instance["system_instruction"] = chosen_sys_instruction
            instances.append(instance)
    elif "flan_zsopt_data" in filepath:
        all_indices = MAPPING_ORCA_TASK_INDEX["flan"]
        all_system_prompts = [ORCA_SYSTEM_PROMPTS[index] for index in all_indices]
        subset_indices = list(filter(lambda x: x != 7 and x != 9, all_indices))
        subset_system_prompts = [ORCA_SYSTEM_PROMPTS[index] for index in subset_indices]
        print(f"all_sys prompts: {len(all_system_prompts)} | subset_system_prompts: {len(subset_system_prompts)}")
        for instance in tqdm(data):
            instance = eval(instance)
            instruction = instance["instruction"]
            if "options:" in instruction or "Options:" in instruction:
                # print(1111)
                chosen_sys_instruction = random.choice(all_system_prompts)
            else:
                # print(2222)
                chosen_sys_instruction = random.choice(subset_system_prompts)
            instance["system_instruction"] = chosen_sys_instruction
            instances.append(instance)

    with open(os.path.join(save_root, filepath), "a") as f:
        for instance in tqdm(instances):
            f.write(json.dumps(instance))
            f.write("\n")
