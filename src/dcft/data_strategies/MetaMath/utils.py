import sys

from MetaMath.code_for_generating_data.code.main_create_backward_questions import MATH, GSM8K
from MetaMath.code_for_generating_data.code.main_forward_reasoning import SCComplexCoT
from MetaMath.code_for_generating_data.code.main_backward_reasoning import BackwardReasoning
from MetaMath.code_for_generating_data.code.main_rephrase_question import RephraseQuestion
from MetaMath.code_for_generating_data.code.main_self_verification_new import SelfVerification

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
    num_repeat: int = 1
    batch_size: int = 500
    time_out: int = 10000
    num_proc: int = 16


def generate_backwards_questions() -> Dataset:
    sys.path.append("dcft/external_repositories/MetaMath/code_for_generating_data/code")
    args = Config()
    math_dataset = MATH(args).make_inv_question()
    gsm8k_dataset = GSM8K(args).make_inv_question()
    mathdf = pd.DataFrame(math_dataset)
    gsm8kdf = pd.DataFrame(gsm8k_dataset)
    dataset = concatenate_datasets([Dataset.from_pandas(mathdf), Dataset.from_pandas(gsm8kdf)])
    return dataset


def generate_input_examples() -> Dataset:
    sys.path.append("dcft/external_repositories/MetaMath/code_for_generating_data/code")
    args = Config()
    math_dataset = MATH(args).get_input_data()
    gsm8k_dataset = GSM8K(args).get_input_data()
    mathdf = pd.DataFrame(math_dataset)
    gsm8kdf = pd.DataFrame(gsm8k_dataset)
    dataset = concatenate_datasets([Dataset.from_pandas(mathdf), Dataset.from_pandas(gsm8kdf)])
    return dataset


def generate_forward_questions(dataset: Dataset) -> Dataset:
    sys.path.append("dcft/external_repositories/MetaMath/code_for_generating_data/code")
    args = Config()
    all_examples = []
    for method in ["GSM8K", "MATH"]:
        args.ds = method
        method = SCComplexCoT(args, dataset)
        all_examples.extend(method.fetch_data_from_openai())

    df = pd.DataFrame(all_examples)
    dataset = Dataset.from_pandas(df)
    return dataset


def rephrase_questions(dataset: Dataset) -> Dataset:
    sys.path.append("dcft/external_repositories/MetaMath/code_for_generating_data/code")
    args = Config()
    all_examples = []
    for method in ["GSM8K", "MATH"]:
        args.ds = method
        args.num_repeat = 1
        method = RephraseQuestion(args, dataset)
        all_examples.extend(method.fetch_data_from_openai())

    df = pd.DataFrame(all_examples)

    dataset = Dataset.from_pandas(df)

    return dataset


def self_verify(dataset: Dataset) -> Dataset:
    sys.path.append("dcft/external_repositories/MetaMath/code_for_generating_data/code")
    args = Config()
    all_examples = []
    for method in ["GSM8K", "MATH"]:
        args.ds = method
        rephrase_cot = SelfVerification(args, dataset)
        all_examples.extend(rephrase_cot.fetch_data_from_openai())

    df = pd.DataFrame(all_examples)

    dataset = Dataset.from_pandas(df)
    return dataset


def backwards_reason(inv_q_dataset: Dataset, original_dataset: Dataset, method_name: str) -> Dataset:
    sys.path.append("dcft/external_repositories/MetaMath/code_for_generating_data/code")
    args = Config()
    all_examples = []
    for method in ["GSM8K", "MATH"]:
        args.method_name = method_name
        args.ds = method
        backwards_reason = BackwardReasoning(args, inv_q_dataset, original_dataset)
        all_examples.extend(backwards_reason.fetch_data_from_openai())

    df = pd.DataFrame(all_examples)
    dataset = Dataset.from_pandas(df)
    return dataset
