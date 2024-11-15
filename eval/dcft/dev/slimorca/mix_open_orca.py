from datasets import load_dataset, concatenate_datasets

ds_names = [
    "mlfoundations-dev/open-orca-cot-judged",
    "mlfoundations-dev/open-orca-niv2-judged",
    "mlfoundations-dev/open-orca-flan-judged",
    "mlfoundations-dev/open-orca-t0-judged",
]

all_ds = []
for ds_name in ds_names:
    ds = load_dataset(ds_name, split="train")
    print(ds)
    all_ds.append(ds)

dataset = concatenate_datasets(all_ds)
print(dataset)
dataset.push_to_hub("mlfoundations-dev/open-orca")
