from dcft.dataset.hf._basedataset import BaseFTDataset
from tqdm import tqdm

class CodeGlaiveFTDataset(BaseFTDataset):
    def __init__(self, data):
        super().__init__(data)
        
        print("Loading and reformatting CodeGlaive dataset")
        for d in tqdm(self.data['train']):
            self.system_prompts.append("")
            self.user_prompts.append(self.reformat_glaive(d['question']))
            self.annotations_original.append(d['answer'])

    def reformat_glaive(self, instruction):
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:"
        )
            