from bespokelabs.curator import LLM
from datasets import load_dataset


class Answer(LLM):
    def prompt(self, row):
        return row["context"]

    def parse(self, row, response):
        row["model_outputs"] = response
        return row


for ds_name in [
    "AIME24_evalchemy",
    "AIME25_evalchemy",
    "AMC23_evalchemy",
    "MATH500_evalchemy",
    "LiveCodeBench_evalchemy",
    "GPQADiamond_evalchemy",
]:
    ds = load_dataset(f"mlfoundations-dev/{ds_name}", split="train")
    answer = Answer(model_name="gpt-4o-mini")
    ds = answer(ds)
    ds.push_to_hub(f"mlfoundations-dev/{ds_name}_gpt-4o-mini")
