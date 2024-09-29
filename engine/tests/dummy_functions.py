import logging

from datasets import Dataset


def dummy_function1(dataset: Dataset, **kwargs) -> Dataset:
    print("Dummy function 1 called")
    return dataset

def dummy_function2(dataset: Dataset, **kwargs) -> Dataset:
    print("Dummy function 2 called")
    return dataset

def dummy_uppercase(dataset: Dataset, **kwargs) -> Dataset:
    return dataset.map(lambda x: {**x, 'output': x['output'].upper()})

def dummy_add_exclamation(dataset: Dataset, **kwargs) -> Dataset:
    return dataset.map(lambda x: {**x, 'output': x['output'] + '!'})