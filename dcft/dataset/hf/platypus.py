from dcft.dataset.hf._basedataset import BaseFTDataset
from tqdm import tqdm


class PlatypusFTDataset(BaseFTDataset):
    def __init__(self, data):
        super().__init__(data)

        print(f"Loading and reformatting {self.__class__.__name__} dataset")
        for d in tqdm(self.data["train"]):
            self.system_prompts.append("")
            self.user_prompts.append(self.reformat(d["instruction"], d["input"]))
            self.annotations_original.append(d["output"])

    def reformat(self, instruction, input):
        if input is None or len(input) == 0:
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:"
            )
        else:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            )
