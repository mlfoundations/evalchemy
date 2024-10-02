from dcft.external_repositories.MetaMath.code_for_generating_data.code.main_create_backwards_question import MATH, GSM8K
from datasets import Dataset, concatenate_datasets

def generate_backwards_questions() -> Dataset:
    math_dataset = MATH().make_inv_question()
    gsm8k_dataset = GSM8K().make_inv_question()
    return concatenate_datasets([math_dataset, gsm8k_dataset])

def generate


