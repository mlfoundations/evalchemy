from dcft.dataset.hf._basedataset import BaseFTDataset
from tqdm import tqdm
from typing import Optional, Any


class AlpacaFTDataset(BaseFTDataset):
    def __init__(self, data: Any) -> None:
        super().__init__(data)

        print(f"Loading and reformatting {self.__class__.__name__} dataset")
        for d in tqdm(self.data["train"]):
            self.system_prompts.append("")
            self.user_prompts.append(self.reformat_alpaca(d["instruction"], d["input"]))
            self.annotations_original.append(d["output"])

    def reformat_alpaca(self, instruction: str, input: Optional[str]) -> str:
        """
        Reformats an instruction and input into a user prompt in the Alpaca format.

        Args:
            instruction (str): The instruction to be reformatted.
            input (Optional[str]): The input providing further context for the instruction.

        Returns:
            str: A formatted user prompt in the Alpaca style.
        """
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
