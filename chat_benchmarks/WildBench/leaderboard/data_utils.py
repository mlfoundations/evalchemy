from datasets import load_dataset, Dataset
import os 
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar 

import random 
disable_progress_bar()
import math 
import json 
from tqdm import tqdm
import numpy as np

id_to_data = None 
model_len_info = None 
bench_data = None 
eval_results = None 
score_eval_results = None 
BASE_SCORE_RESULTS_PATH = "eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13/"
BASE_EVAL_RESULTS_PATH = "eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/"


task_group_new = {
    "Information seeking": "Information/Advice seeking",
    "Creative Writing": "Creative Tasks",
    "Coding & Debugging": "Coding & Debugging",
    "Reasoning": "Planning & Reasoning",
    "Editing": "Creative Tasks",
    "Math": "Math & Data Analysis",
    "Planning": "Planning & Reasoning",
    "Brainstorming": "Creative Tasks",
    "Role playing": "Creative Tasks",
    "Advice seeking": "Information/Advice seeking",
    "Data Analysis": "Math & Data Analysis",
    "Others": "Creative Tasks"
}

# Formats the columns
def formatter(x):
    if type(x) is str:
        x = x
    else: 
        x = round(x, 1)
    return x
 
 
def load_benchdata():
    global bench_data, id_to_data
    print("Loading WildBench data...")
    if bench_data is None:
        bench_data = load_dataset("WildEval/WildBench-V2", "v2.0522", split="test")
    return bench_data

def load_benchdata_dict():
    global bench_data, id_to_data
    # print("Loading WildBench data....")
    if bench_data is None:
        bench_data = load_benchdata()
    if id_to_data is None:
        id_to_data = {}
        for item in bench_data:
            id_to_data[item["session_id"]] = item
    return id_to_data

def load_eval_results():
    global eval_results, score_eval_results
    # print("Loading WildBench Evaluation data...")
    # Go through the eval results folder "WildBench-main/eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09"
    
    eval_results = {}
    score_eval_results = {}

    for file in os.listdir(BASE_SCORE_RESULTS_PATH):
        if file.endswith(".json"):
            with open(os.path.join(BASE_SCORE_RESULTS_PATH, file), "r") as f:
                model_name = file.replace(".json", "").replace("@together", "")
                score_eval_results[model_name] = json.load(f)
    
    sub_dirs = ["ref=gpt-4-turbo-2024-04-09", "ref=claude-3-haiku-20240307", "ref=Llama-2-70b-chat-hf"]
    for sub_dir in sub_dirs:
        eval_results[sub_dir] = {}
        path = os.path.join(BASE_EVAL_RESULTS_PATH, sub_dir)
        for file in os.listdir(path):
            if file.endswith(".json"):
                with open(os.path.join(path, file), "r") as f:
                    model_name = file.replace(".json", "").replace("@together", "")
                    eval_results[sub_dir][model_name] = json.load(f)
    # print(eval_results.keys())
    # print(eval_results[sub_dirs[0]].keys())
    # print(score_eval_results.keys())
    return eval_results, score_eval_results

def load_infer_results(model_name):
    # print(f"Loading WildBench Results for {model_name}...")
    # infer_results = load_dataset("WildEval/WildBench-Results", model_name, split="train")
    bench_data = load_dataset("WildEval/WildBench-Results-V2.0522", model_name, split="train")
    return bench_data



def sample_an_eval_result(model_list=[], tag_list=[], eval_mode="score", sample_session_id=None, return_all=False):
    global id_to_data, eval_results, score_eval_results

    # print the args 
    print(f"Model List: {model_list} | Tag List: {tag_list} | Eval Mode: {eval_mode} | Sample Session ID: {sample_session_id}")

    if eval_results is None:
        eval_results, score_eval_results = load_eval_results()
    if id_to_data is None:
        id_to_data = load_benchdata_dict()       
    
    all_valid_results = []
    if eval_mode == "score":
        if len(model_list) < 2:
            # random add models to at least 2
            model_list = model_list + random.sample(list(score_eval_results.keys()), 2 - len(model_list))
        random_model_A = random.choice(model_list)
        random_model_B = random.choice(model_list)
        while random_model_A == random_model_B:
            random_model_B = random.choice(model_list)
        formatted_eval_results = []
        A_data_by_id = {}
        B_data_by_id = {}
        print(score_eval_results.keys())
        for item in score_eval_results[random_model_A]:
            A_data_by_id[item["session_id"]] = item
        for item in score_eval_results[random_model_B]:
            B_data_by_id[item["session_id"]] = item
        # intersection of both ids
        common_ids = set(A_data_by_id.keys()).intersection(set(B_data_by_id.keys()))
        # shuffle the ids 
        common_ids = list(common_ids)
        random.shuffle(common_ids)
        # random select a common id, whose task type is in tag_list
        if sample_session_id and sample_session_id in common_ids:
            common_ids = [sample_session_id]
        for session_id in common_ids:
            data_item = id_to_data[session_id]
            item_A = A_data_by_id[session_id]
            item_B = B_data_by_id[session_id]
            task_type = task_group_new[data_item['primary_tag']]
            task_tags = [task_group_new[data_item['primary_tag']]] + [task_group_new[x] for x in data_item['secondary_tags']]
            #     continue
            if tag_list and task_type not in tag_list:
                continue  

            conversation_input = data_item["conversation_input"] 
            score_A = item_A["score"]
            score_B = item_B["score"]
            reasons_A = item_A["parsed_result"]
            reasons_B = item_B["parsed_result"]
            reason_all = {
                "Model A's Strengths": reasons_A["strengths"],
                "Model A's Weaknesses": reasons_A["weaknesses"],
                "Model A's score": score_A,
                "Model B's Strengths": reasons_B["strengths"],
                "Model B's Weaknesses": reasons_B["weaknesses"],
                "Model B's score": score_B,
            }
            if int(score_A) > int(score_B):
                winner = random_model_A
            elif int(score_A) < int(score_B):
                winner = random_model_B
            else:
                winner = "Tie"

            result_item = {
                "session_id": session_id, 
                "intent": data_item["intent"],
                "task_type": task_type,
                "task_tags": task_tags,
                "conversation_input": conversation_input, 
                "checklist": data_item["checklist"],
                "model_A": random_model_A,
                "model_B": random_model_B,
                "model_A_output": item_A["model_output"],
                "model_B_output": item_B["model_output"],
                "winner": winner,
                "parsed_result": reason_all,
                "choice": winner,
                
            }
            if return_all is False:
                return result_item
            else:
                all_valid_results.append(result_item)
    else:
        # random select a model from model_list
        random_model_name = random.choice(model_list)
        formatted_eval_results = []  
        # print(eval_results[eval_mode].keys())
        for item in eval_results[eval_mode][random_model_name]: 
            session_id = item["session_id"]
            if sample_session_id and session_id != sample_session_id:
                continue
            result_item = {
                "session_id": item["session_id"],
                "model_A": item["model_A"].split("/")[-1],
                "model_B": item["model_B"].split("/")[-1],
                "model_A_output": item["model_outputs"][item["model_A"]],
                "model_B_output": item["model_outputs"][item["model_B"]],
                "winner": item["winner"],
                "parsed_result": item["parsed_result"],
            }
            formatted_eval_results.append(result_item)  
        
        random.shuffle(formatted_eval_results)
        for eval_item in formatted_eval_results:  
            session_id = eval_item['session_id']
            data_item = id_to_data[session_id] 
            model_A = eval_item['model_A']
            model_B = eval_item['model_B']
            winner = eval_item['winner']
            # print(f"## Model A: {model_A} | Model B: {model_B} | Winner: {winner}") 
            if model_list and (model_A not in model_list and model_B not in model_list):
                print(f"Skipping {model_A} and {model_B} as they are not in the model list")
                continue

            task_type = task_group_new[data_item['primary_tag']] # primary task type  
            task_tags = [task_group_new[data_item['primary_tag']]] + [task_group_new[x] for x in data_item['secondary_tags']]
            #     continue
            if tag_list and task_type not in tag_list:
                # print(task_type)
                continue
            
            conversation_input = data_item["conversation_input"] 
            result_dict = eval_item.copy()
            result_dict.update({
                "session_id": eval_item['session_id'], 
                "model_A": model_A,
                "model_B": model_B,
                "winner": winner,
                "intent": data_item["intent"],
                "task_type": task_type,
                "task_tags": task_tags,
                "conversation_input": conversation_input, 
                "reason": eval_item['parsed_result'],
                "choice": eval_item['parsed_result']["choice"],
                "checklist": data_item["checklist"],
            })
            if return_all is False:
                return result_dict
            else:
                all_valid_results.append(result_dict)
    if return_all is True:
        return all_valid_results
    return None 

# id_to_data = load_benchdata_dict()

# main 
if __name__ == "__main__":
    # test the function for sample_an_eval_result 
    # print(sample_an_eval_result(model_list=["Llama-3-Instruct-8B-SimPO"], tag_list=["Planning & Reasoning"], eval_mode="ref=gpt-4-turbo-2024-04-09"))
    print(sample_an_eval_result(model_list=["Llama-3-Instruct-8B-SimPO"], tag_list=['Creative Tasks', 'Planning & Reasoning', 'Math & Data Analysis', 'Information/Advice seeking', 'Coding & Debugging'], eval_mode="ref=claude-3-haiku-20240307"))
    # print(json.dumps(sample_an_eval_result(model_list=["Llama-3-Instruct-8B-SimPO"], tag_list=[], eval_mode="score"), indent=2))