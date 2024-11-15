from datasets import load_dataset, Dataset


def load_and_filter_dataset(dataset_name: str, filter_column: str) -> Dataset:
    # Assumes filter column is boolean
    dataset = load_dataset(dataset_name, split="train")
    num_before = len(dataset)
    dataset = dataset.filter(lambda x: x[filter_column])
    num_after = len(dataset)
    percentage_removed = (num_before - num_after) / num_before * 100
    print(f"Filtered {num_before - num_after} out of {num_before} samples, removing ({percentage_removed:.2f}%)")
    return dataset


print(load_and_filter_dataset("mlfoundations-dev/open-orca-cot-judged", "model_judgement"))
