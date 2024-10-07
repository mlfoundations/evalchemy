import json 
from datasets import load_dataset, Dataset
import os 
import sys


DATA_NAME = "wild_bench_v2"
HF_RESULTS_PATH = "allenai/WildBench-V2-Model-Outputs"

print(f"Uploading to {HF_RESULTS_PATH} with data name {DATA_NAME}.")

def load_and_upload(model_name):
    filepath = f"result_dirs/{DATA_NAME}/{model_name}.json"
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {model_name} results with {len(data)} samples.")
    if len(data) != 1024:
        print(f"Expected 1024 samples, got {len(data)}. Exit!")
        return
    print(f"Uploading {model_name} results to {HF_RESULTS_PATH}... # of examples = {len(data)}")
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(
        repo_id=HF_RESULTS_PATH,
        config_name=model_name,
        split="train",
        token=os.environ.get("HUGGINGFACE_TOKEN"),
        commit_message=f"Add {model_name} results.",
    )
    print(f"Uploaded {model_name} results.")

# load_and_upload(sys.argv[1])

if len(sys.argv) > 1:
    model_name = sys.argv[1]
    load_and_upload(model_name)
else:
    json_files = [
        # "claude-3-haiku-20240307",
        # "claude-3-sonnet-20240229",
        # "dbrx-instruct@together",
        # "gemma-2b-it",
        # "gemma-7b-it",
        # "gpt-3.5-turbo-0125",
        # "gpt-4-turbo-2024-04-09",
        # "gpt-4o-2024-05-13",
        # "Llama-2-7b-chat-hf",
        # "Llama-2-13b-chat-hf",
        # "Llama-2-70b-chat-hf",
        # "Meta-Llama-3-8B-Instruct",
        # "Meta-Llama-3-70B-Instruct",
        # "Mistral-7B-Instruct-v0.2",
        # "mistral-large-2402",
        # "Mixtral-8x7B-Instruct-v0.1",
        # "Nous-Hermes-2-Mixtral-8x7B-DPO",
        # "Qwen1.5-7B-Chat@together",
        # "Qwen1.5-72B-Chat",
        # "Starling-LM-7B-beta",
        # "tulu-2-dpo-70b",
        # "Yi-1.5-34B-Chat",
        # "Yi-1.5-9B-Chat",
        # "Yi-1.5-6B-Chat",        
    ]

    for model_name in json_files:
        load_and_upload(model_name)




""" 
# V2 
python src/upload_results.py  claude-3-opus-20240229
python src/upload_results.py  claude-3-sonnet-20240229
python src/upload_results.py  claude-3-haiku-20240307
python src/upload_results.py  Meta-Llama-3-8B-Instruct
python src/upload_results.py  Meta-Llama-3-70B-Instruct
python src/upload_results.py  gpt-3.5-turbo-0125
python src/upload_results.py  gpt-4-turbo-2024-04-09
python src/upload_results.py  gpt-4o-2024-05-13
python src/upload_results.py  Qwen1.5-72B-Chat
python src/upload_results.py  Qwen1.5-7B-Chat@together
python src/upload_results.py  Yi-34B-Chat
python src/upload_results.py  Mixtral-8x7B-Instruct-v0.1
python src/upload_results.py  Mistral-7B-Instruct-v0.2
python src/upload_results.py  Llama-2-7b-chat-hf
python src/upload_results.py  Llama-2-13b-chat-hf
python src/upload_results.py  Llama-2-70b-chat-hf
python src/upload_results.py  gemma-2b-it
python src/upload_results.py  gemma-7b-it
python src/upload_results.py  mistral-large-2402
python src/upload_results.py  command-r
python src/upload_results.py  gpt-4-0125-preview 

# 2024-05-30
python src/upload_results.py  command-r-plus
python src/upload_results.py  Hermes-2-Theta-Llama-3-8B
python src/upload_results.py  Llama-3-Instruct-8B-SimPO
python src/upload_results.py  Phi-3-mini-128k-instruct
python src/upload_results.py  Phi-3-medium-128k-instruct

"""