from dcft.dataset.hf._basedataset import BaseFTDataset
from tqdm import tqdm
from typing import Any, Dict


class CodeGlaiveFTDataset(BaseFTDataset):
    def __init__(self, data: Dict[str, Any]) -> None:
        """
        Initializes the CodeGlaiveFTDataset class.

        Args:
            data (Dict[str, Any]): The dataset containing training data.
        """
        super().__init__(data)

        print("Loading and reformatting CodeGlaive dataset")
        for d in tqdm(self.data["train"]):
            self.system_prompts.append("")
            self.user_prompts.append(self.reformat_glaive(d["question"]))
            self.annotations_original.append(d["answer"])

    def reformat_glaive(self, instruction: str) -> str:
        """
        Reformats the given instruction into a structured prompt similar to Alpaca

        Args:
            instruction (str): The instruction to be reformatted.

        Returns:
            str: A formatted string that includes the instruction and a placeholder for the response.
        """
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:"
        )
