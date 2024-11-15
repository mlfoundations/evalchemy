from datasets import load_dataset

dataset_names = [
    # "mlfoundations-dev/open-orca-cot-judged",
    # "mlfoundations-dev/open-orca-niv2-judged",
    "mlfoundations-dev/open-orca-flan-judged",
]

for dataset_name in dataset_names:
    ds = load_dataset(dataset_name, split="train")
    print(ds)

    def determine_judgment(sample):
        assistant_message = sample["model_judgement_full"]
        decision_word = assistant_message.strip().lower().split()[-1]
        decision_word = "".join(char for char in decision_word if char.isalpha())
        decision = decision_word == "yes"
        sample["model_judgement"] = decision
        if decision_word not in ["yes", "no"]:
            print(f"WARNING: Defaulting to False for classification '{decision_word}'")

    mapped_ds = ds.map(determine_judgment)
    mapped_ds.push_to_hub(dataset_name)
