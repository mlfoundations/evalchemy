import os 
import sys 
from tqdm import tqdm
# for all files in the folder
BASE_EVAL_RESULTS_PATH = "evaluation/results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/"
sub_dirs = ["ref=gpt-4-turbo-2024-04-09", "ref=claude-3-haiku-20240307", "ref=Llama-2-70b-chat-hf"]

for sub_dir in sub_dirs:
    folder = BASE_EVAL_RESULTS_PATH + sub_dir
    files = os.listdir(folder)
    for filepath in tqdm(files, desc=f"Processing {sub_dir}"):
        print(f"Processing file {filepath}")
        if ".batch_results.jsonl" not in filepath:
            continue
        submit_path = filepath.replace(".batch_results.jsonl", ".batch-submit.jsonl")
        submit_path = os.path.join(folder, submit_path)
        filepath = os.path.join(folder, filepath)
        os.system(f"python src/openai_batch_eval/batch_results_format.py {submit_path} {filepath}")
        print(f"Processed output file {filepath}")
        print("-"*80)

"""
cp evaluation/results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09/*.json eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09/
cp evaluation/results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=Llama-2-70b-chat-hf/*.json eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=Llama-2-70b-chat-hf/
cp evaluation/results_v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307/*.json eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307/
cp evaluation/results_v2.0522/score.v2/eval=gpt-4-turbo-2024-04-09/*.json eval_results/v2.0522/score.v2/eval=gpt-4-turbo-2024-04-09/
"""