from datasets import concatenate_datasets
from dcft.data_strategies.SlimOrca.utils import load_and_filter_dataset, load_dataset

ds_names = [
    "mlfoundations-dev/open-orca-cot-judged",
    "mlfoundations-dev/open-orca-niv2-judged",
    "mlfoundations-dev/open-orca-flan-judged",
    "mlfoundations-dev/open-orca-t0-judged",
]

all_ds = []
for ds_name in ds_names:
    ds = load_and_filter_dataset(ds_name, filter_column="model_judgement")
    print(ds)
    all_ds.append(ds)

dataset = concatenate_datasets(all_ds)
# dataset = load_dataset("mlfoundations-dev/slim-orca", split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(517_982))
print(dataset)
dataset.push_to_hub("mlfoundations-dev/slim-orca-500k")
