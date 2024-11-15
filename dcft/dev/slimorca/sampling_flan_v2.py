import os
import argparse
import random
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser(description="sampling flanv2 instructions for slimorca reproduction")
parser.add_argument(
    "--subset",
    type=str,
    default="cot_zsopt_data",
)
args = parser.parse_args()


def collect_all_data(root, folder):
    folder_data = []
    for filename in tqdm(os.listdir(os.path.join(root, folder))):
        filename = os.path.join(root, folder, filename)
        with open(filename, "r") as f:
            data = list(f)
        folder_data = folder_data + data
    print(f"length of folder data: {len(folder_data)}")
    return folder_data


def get_task_index_mapping(data):
    res = defaultdict(list)
    for j in tqdm(range(len(data))):
        example = eval(data[j])
        metadata = example["metadata"]
        res[metadata["_task_name"]].append(j)
    return res


def get_stratified_sampling(folder_data, task_index_mapping, source):
    subsampled_data = []
    if source == "niv2_zsopt_data":
        ## 300 queries per task
        for task in tqdm(task_index_mapping):
            indices = task_index_mapping[task]
            indices = random.sample(indices, 300)
            for index in indices:
                subsampled_data.append(folder_data[index])
    elif source == "flan_zsopt_data" or source == "t0_zsopt_data":
        budget = 2.5e6 if source == "flan_zsopt_data" else 2e6
        print(f"budget: {budget}")
        count = 0
        tasks = list(task_index_mapping.keys())
        used_index = set()
        while count < budget:
            if count != 0 and count % 1e4 == 0:
                print(f"count: {count}")
            task = random.choice(tasks)
            indices = task_index_mapping[task]
            if len(indices):
                index = random.choice(indices)
                if index not in used_index:
                    subsampled_data.append(folder_data[index])
                    count += 1
                    ## without replacement
                    used_index.add(index)
                    task_index_mapping[task] = indices
    else:
        subsampled_data = []
        print("Not implemented")
    return subsampled_data


def save_data(data, save_folder, folder):
    with open(os.path.join(save_folder, f"{folder}.jsonl"), "a") as f:
        for instance in tqdm(data):
            f.write(instance)
    print(f"done {folder}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the data")
    parser.add_argument("--save_folder", type=str, required=True, help="Directory to save the subsampled data")
    args = parser.parse_args()

    root = args.root
    save_folder = args.save_folder
    folder = args.subset

    if folder == "cot_zsopt_data":
        folder_data = collect_all_data(root, folder)
        save_data(folder_data, save_folder, folder)

    if folder == "niv2_zsopt_data" or folder == "flan_zsopt_data" or folder == "t0_zsopt_data":
        folder_data = collect_all_data(root, folder)
        task_index_mapping = get_task_index_mapping(folder_data)
        print(len(task_index_mapping))
        subsampled_data = get_stratified_sampling(folder_data, task_index_mapping, source=folder)
        print(f"length of subsampled data: {len(subsampled_data)}")
        save_data(subsampled_data, save_folder, args.subset)

    print(f"done {folder}")


if __name__ == "__main__":
    main()
