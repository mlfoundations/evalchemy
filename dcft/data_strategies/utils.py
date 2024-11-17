from datasets import Dataset


def hf_upload(dataset: Dataset, dataset_name: str, hf_account: str = "mlfoundations-dev", hf_private: bool = False):
    repo_id = f"{hf_account}/{dataset_name}"
    commit_message = f"Uploading {repo_id}"
    dataset.push_to_hub(repo_id=repo_id, commit_message=commit_message, private=hf_private)
