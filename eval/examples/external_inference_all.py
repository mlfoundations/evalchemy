from bespokelabs.curator import LLM
from datasets import load_dataset

ds = load_dataset("mlfoundations-dev/REASONING_evalchemy", split="train")


class Answer(LLM):
    def prompt(self, row):
        return row["context"]

    def parse(self, row, response):
        row["model_outputs"] = response
        return row


answer = Answer(model_name="gpt-4o-mini")
ds = answer(ds)
ds.push_to_hub("mlfoundations-dev/REASONING_evalchemy_gpt-4o-mini")
print("Viewable at https://huggingface.co/datasets/mlfoundations-dev/REASONING_evalchemy_gpt-4o-mini")
