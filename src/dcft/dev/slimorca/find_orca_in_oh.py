from datasets import Dataset, load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    # Load ORCA
    orca = load_dataset("Open-Orca/OpenOrca")
    orca_questions = orca["train"]["question"]

    # Load OH and extract null rows
    openhermes = load_dataset("teknium/OpenHermes-2.5")
    openhermes_df = openhermes["train"].to_pandas()
    openhermes_df = openhermes_df[openhermes_df["source"].isna()]
    openhermes_null = Dataset.from_pandas(openhermes_df)

    openhermes_null_questions = []
    for convo in tqdm(openhermes_null["conversations"]):
        # Extract first message from human
        for message in convo:
            if message["from"] == "human":
                openhermes_null_questions.append(message["value"])
                break

    overlap = set(openhermes_null_questions) & set(orca_questions)
    still_missing = set(openhermes_null_questions) - set(orca_questions)

    print("Number of overlaps: ", len(overlap))
    print("Number of still missing: ", len(still_missing))
