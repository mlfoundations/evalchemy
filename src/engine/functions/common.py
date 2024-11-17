from datasets import load_dataset, Dataset

## CONVERSIONS BETWEEN DATA FORMATS ##


def load_IT_dataset(
    dataset_name: str, instruction_column: str, response_column: str, drop_other_columns: bool = True
) -> Dataset:
    dataset = load_dataset(dataset_name, split="train")
    if drop_other_columns:
        cols_to_remove = dataset.column_names
        if instruction_column not in cols_to_remove:
            raise ValueError(f"Instruction column {instruction_column} not found in dataset {dataset_name}")
        if response_column not in cols_to_remove:
            raise ValueError(f"Response column {response_column} not found in dataset {dataset_name}")
        cols_to_remove.remove(instruction_column)
        cols_to_remove.remove(response_column)
        dataset = dataset.remove_columns(cols_to_remove)

    if instruction_column != "instruction":
        dataset = dataset.rename_column(instruction_column, "instruction")
    if response_column != "response":
        dataset = dataset.rename_column(response_column, "response")
    new_column = [dataset_name] * len(dataset)
    dataset = dataset.add_column("source", new_column)
    return dataset


def load_IT_from_alpaca_format(dataset_name: str) -> Dataset:
    dataset = load_dataset(dataset_name, split="train")

    def load_IT_from_sample(sample):
        instruction = sample["instruction"]
        input = sample["input"]
        output = sample["output"]
        instruction = instruction + "\n\n" + input
        return {"instruction": instruction, "response": output}

    dataset = dataset.map(load_IT_from_sample)
    cols_to_remove = dataset.column_names
    cols_to_remove.remove("instruction")
    cols_to_remove.remove("response")
    dataset = dataset.remove_columns(cols_to_remove)
    new_column = [dataset_name] * len(dataset)
    dataset = dataset.add_column("source", new_column)
    return dataset


def load_IT_from_unnatural_instructions_format(dataset_name: str) -> Dataset:
    dataset = load_dataset(dataset_name, split="train")

    def load_IT_from_sample(sample):
        reformulations = sample["reformulations"][0]
        instruction = reformulations["instruction"]
        input = reformulations["input"]
        output = reformulations["output"]
        instruction = instruction.replace("{INPUT}", f'"{input}"')
        return {"instruction": instruction, "response": output}

    dataset = dataset.map(load_IT_from_sample)

    cols_to_remove = dataset.column_names
    cols_to_remove.remove("instruction")
    cols_to_remove.remove("response")
    dataset = dataset.remove_columns(cols_to_remove)
    new_column = [dataset_name] * len(dataset)
    dataset = dataset.add_column("source", new_column)
    return dataset


def convert_IT_to_ShareGPT_format(
    dataset: Dataset, instruction_column: str = "instruction", response_column: str = "response"
) -> Dataset:

    def sharegpt_from_it(sample):
        instruction = {"from": "human", "value": sample[instruction_column]}
        response = {"from": "gpt", "value": sample[response_column]}
        return {"conversations": [instruction, response]}

    dataset = dataset.map(sharegpt_from_it)

    return dataset


def load_ShareGPT_dataset_as_IT(dataset_name: str, truncate: int = None) -> Dataset:
    dataset = load_dataset(dataset_name, split="train")
    if truncate is not None:
        dataset = dataset.select(range(truncate))
    return convert_ShareGPT_to_IT_format(dataset)


def convert_ShareGPT_to_IT_format(dataset: Dataset) -> Dataset:
    def it_from_sharegpt(sample):
        if sample["conversations"][0]["from"] == "human":
            instruction = sample["conversations"][0]["value"]
            assert sample["conversations"][1]["from"] == "gpt"
            response = sample["conversations"][1]["value"]
        elif sample["conversations"][1]["from"] == "human":
            # sometimes the first message is system instructions - ignoring them here
            # specifically in OH, the system instructions are only present for airoboros2.2 and slimorca
            # airoboros2.2 provides character cards or "you are a trivia AI"
            # slimorca provides CoT instructions a la "you are a helpful assistant and explain your steps"
            instruction = sample["conversations"][1]["value"]
            assert sample["conversations"][2]["from"] == "gpt"
            response = sample["conversations"][2]["value"]
        else:
            raise ValueError("Invalid conversation format")
        return {"instruction": instruction, "original_response": response}

    dataset = dataset.map(it_from_sharegpt)
    dataset = dataset.remove_columns(["conversations"])
    dataset = dataset.select_columns(["instruction", "original_response"])
    return dataset
