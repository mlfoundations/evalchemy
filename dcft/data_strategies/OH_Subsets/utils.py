from datasets import Dataset


def filter_away_sources(dataset_with_sources: Dataset, actual_dataset: Dataset, source_to_remove: str) -> Dataset:

    instruction_to_source_mapping = {}
    for row in dataset_with_sources:
        if row["conversations"][0]["from"] == "human":
            instruction_to_source_mapping[row["conversations"][0]["value"]] = row["source_label_exact"]
        else:
            instruction_to_source_mapping[row["conversations"][1]["value"]] = row["source_label_exact"]

    all_examples = []
    num_missing = 0
    for i in range(len(actual_dataset)):
        if actual_dataset[i]["conversations"][0]["from"] == "human":
            if actual_dataset[i]["conversations"][0]["value"] in instruction_to_source_mapping:
                source_label = instruction_to_source_mapping[actual_dataset[i]["conversations"][0]["value"]]
                if source_to_remove not in source_label:
                    all_examples.append(
                        {"conversations": actual_dataset[i]["conversations"], "source_label_exact": source_label}
                    )
            else:
                num_missing += 1
        else:
            if actual_dataset[i]["conversations"][1]["value"] in instruction_to_source_mapping:
                source_label = instruction_to_source_mapping[actual_dataset[i]["conversations"][1]["value"]]
                if source_to_remove not in source_label:
                    all_examples.append(
                        {"conversations": actual_dataset[i]["conversations"], "source_label_exact": source_label}
                    )
            else:
                num_missing += 1
    print(f"Num Missing: {num_missing}")
    actual_dataset = Dataset.from_list(all_examples)
    return actual_dataset
