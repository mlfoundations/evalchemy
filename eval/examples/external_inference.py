from bespokelabs.curator import LLM
from datasets import load_dataset

ds = load_dataset("mlfoundations-dev/AIME24_evalchemy", split="train")


class Answer(LLM):
    def prompt(self, row):
        return row["context"]

    def parse(self, row, response):
        row["model_outputs"] = response
        return row


answer = Answer(model="gpt-4o-mini")
ds = answer.run(ds)
ds.push_to_hub("mlfoundations-dev/AIME24_evalchemy_gpt-4o-mini")
