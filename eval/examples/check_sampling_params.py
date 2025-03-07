import os

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# dataset_name = "mlfoundations-dev/OpenThinker-7B_eval_03-07-25_16-50_2870"
"""
       tokens_count
count    150.000000
mean   14992.933333
std    10779.196801
min      914.000000
25%     7474.500000
50%    11584.500000
75%    18942.500000
max    32701.000000
"""
# dataset_name = "mlfoundations-dev/OpenThinker-7B_eval_03-07-25_09-02_0981"
"""
       tokens_count
count   3127.000000
mean   12057.965782
std    12748.102026
min       38.000000
25%     1217.500000
50%     6018.000000
75%    31697.000000
max    33760.000000
"""
dataset_name = "mlfoundations-dev/Qwen2.5-7B-Instruct_eval_03-07-25_08-20_0981"
"""
       tokens_count
count   3127.000000
mean    4448.372562
std    10684.242403
min        5.000000
25%      128.000000
50%      326.000000
75%      761.000000
max    32745.000000
"""
# generally this looks all right to me....
split = "train"
ds = load_dataset(dataset_name, split=split)
tokenizer = AutoTokenizer.from_pretrained("open-thoughts/OpenThinker-7B", trust_remote_code=True)


def count_tokens(example):
    tokens = len(tokenizer.encode(example["model_outputs"]))
    example["tokens_count"] = tokens
    return example


ds = ds.map(count_tokens, num_proc=os.cpu_count())

# Summarize token counts using pandas
token_counts = [example["tokens_count"] for example in ds]
df = pd.DataFrame(token_counts, columns=["tokens_count"])
print(df.describe())
