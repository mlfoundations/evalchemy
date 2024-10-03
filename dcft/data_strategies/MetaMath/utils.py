from dcft.external_repositories.MetaMath.code_for_generating_data.code.main_create_backwards_question import MATH, GSM8K
from dcft.external_repositories.MetaMath.code_for_generating_data.code.main_forward_reasoning import SCComplexCoT
from datasets import Dataset, concatenate_datasets
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Config:
    eng: str = "gpt-4o-mini"
    ds: str = "GSM8K"
    part: str = ""
    temp: float = 0.7
    method_name: str = "SCComplexCoT"
    cont: bool = False
    num_repeat: int = 10
    batch_size: int = 20
    time_out: int = 30
    num_proc: int = 16

def generate_backwards_questions() -> Dataset:
    math_dataset = MATH().make_inv_question()
    gsm8k_dataset = GSM8K().make_inv_question()
    return concatenate_datasets([math_dataset, gsm8k_dataset])

def generate_forward_questions(dataset: Dataset) -> Dataset:
    args = Config()
    all_examples = []
    for method in ["GSM8K", "MATH"]:
        args.ds = method
        method = SCComplexCoT(args, dataset)
        all_examples.extend(method.fetch_data_from_openai())
    
    df = pd.DataFrame(all_examples)
    
    # Convert pandas DataFrame to Hugging Face Dataset
    return Dataset.from_pandas(df)
    

