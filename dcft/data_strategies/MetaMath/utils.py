import sys
sys.path.append('dcft/external_repositories/MetaMath/code_for_generating_data/code')

from dcft.external_repositories.MetaMath.code_for_generating_data.code.main_create_backward_questions import MATH, GSM8K
from dcft.external_repositories.MetaMath.code_for_generating_data.code.main_forward_reasoning import SCComplexCoT
from dcft.external_repositories.MetaMath.code_for_generating_data.code.main_backward_reasoning import BackwardReasoning
from dcft.external_repositories.MetaMath.code_for_generating_data.code.main_rephrase_question import RephraseQuestion
from dcft.external_repositories.MetaMath.code_for_generating_data.code.main_self_verification import SelfVerification

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
    batch_size: int = 3
    time_out: int = 30
    num_proc: int = 16

def generate_backwards_questions(_: Dataset) -> Dataset:
    
    args = Config()
    math_dataset = MATH(args).make_inv_question()
    gsm8k_dataset = GSM8K(args).make_inv_question()
    mathdf = pd.DataFrame(math_dataset)
    gsm8kdf = pd.DataFrame(gsm8k_dataset)
    dataset = concatenate_datasets([Dataset.from_pandas(mathdf),  Dataset.from_pandas(gsm8kdf)])
    dataset = dataset.select(list(range(30)))
    return dataset

def generate_forward_questions(dataset: Dataset) -> Dataset:
    args = Config()
    all_examples = []
    for method in ["GSM8K", "MATH"]:
        args.ds = method
        method = SCComplexCoT(args, dataset)
        all_examples.extend(method.fetch_data_from_openai())
    
    df = pd.DataFrame(all_examples)
    
    # Convert pandas DataFrame to Hugging Face Dataset
    dataset =  Dataset.from_pandas(df)
    return dataset


def rephrase_questions(dataset: Dataset) -> Dataset:
    args = Config()
    all_examples = []
    for method in ["GSM8K", "MATH"]:
        args.ds = method
        method = RephraseQuestion(args, dataset)
        all_examples.extend(method.fetch_data_from_openai())

    df = pd.DataFrame(all_examples)
    
    dataset =  Dataset.from_pandas(df)
    dataset = dataset.select(list(range(30)))
    return dataset

def self_verify(dataset: Dataset) -> Dataset:
    args = Config()
    all_examples = []
    for method in ["GSM8K", "MATH"]:
        args.ds = method
        rephrase_cot = SelfVerification(args, dataset)
        rephrase_cot.fetch_data_from_openai()
        all_examples.extend(rephrase_cot.fetch_data_from_openai())

    df = pd.DataFrame(all_examples)
    
    dataset =  Dataset.from_pandas(df)
    dataset = dataset.select(list(range(30)))
    return dataset
        