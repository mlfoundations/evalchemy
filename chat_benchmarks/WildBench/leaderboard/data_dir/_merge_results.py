import json 
import os 
import sys 

# try:
#     K = int(sys.argv[1])
# except:
#     print("No K specified, so using K=-1")
#     K = -1

wb_elo_results = {}
with open("leaderboard/data_dir/wb_elo_results.json", "r") as f:
    wb_elo_results = json.load(f)
wb_elo_stat = wb_elo_results["elo_stat"]

def merge_scores(K=-1):
    haiku_rewards_file = f"leaderboard/data_dir/pairwise-haiku-K={K}.json"
    llama_rewards_file = f"leaderboard/data_dir/pairwise-llama-K={K}.json"
    gpt4t_rewards_file = f"leaderboard/data_dir/pairwise-gpt4t-K={K}.json"
     
    score_file = "leaderboard/data_dir/score.json"


    haiku_rewards = {}
    llama_rewards = {}
    gpt4t_rewards = {}
    if os.path.exists(haiku_rewards_file):
        with open(haiku_rewards_file, "r") as f:
            haiku_rewards = json.load(f)
    if os.path.exists(llama_rewards_file):
        with open(llama_rewards_file, "r") as f:
            llama_rewards = json.load(f)
    if os.path.exists(gpt4t_rewards_file):
        with open(gpt4t_rewards_file, "r") as f:
            gpt4t_rewards = json.load(f)

    scores = {}
    with open(score_file, "r") as f:
        scores = json.load(f)

    all_stat = {}
    with open("leaderboard/data_dir/all_stat.json", "r") as f:
        all_stat = json.load(f) 

    missing_models = []
    for model in scores:
        if model not in all_stat:
            missing_models.append(model)

    all_models = list(scores.keys())

    elo_only_models = []

    for model in all_models:
        if model not in all_stat:
            all_stat[model] = {}
            all_stat[model]["Arena Elo (hard-en) - latest"] = "-"
            all_stat[model]["Arena-Hard v0.1"] = "-"
            all_stat[model]["AE2.0 LC"] = "-"
            all_stat[model]["AE2.0"] = "-"
        
        all_stat[model][f"haiku_reward.K={K}"] = H = haiku_rewards.get(model, {"reward": 0})["reward"]*100
        all_stat[model][f"llama_reward.K={K}"] = L = llama_rewards.get(model, {"reward": 0})["reward"]*100
        all_stat[model][f"gpt4t_reward.K={K}"] = G = gpt4t_rewards.get(model, {"reward": 0})["reward"]*100
        
        # all_task_types = ['Information seeking', 'Creative Writing', 'Coding & Debugging', 'Reasoning', 'Editing', 'Math', 'Planning', 'Brainstorming', 'Role playing', 'Advice seeking', 'Data Analysis']
        all_task_types = ['Creative Tasks', 'Planning & Reasoning', 'Math & Data Analysis', 'Information/Advice seeking', 'Coding & Debugging'] # merged version
        for task_tag in all_task_types:
            if model in haiku_rewards:
                H_TAG = haiku_rewards[model]["task_categorized_rewards"][task_tag]*100
            else:
                H_TAG = 0
            if model in llama_rewards:
                L_TAG = llama_rewards[model]["task_categorized_rewards"][task_tag]*100
            else:
                L_TAG = 0
            if model in gpt4t_rewards:
                G_TAG = gpt4t_rewards[model]["task_categorized_rewards"][task_tag]*100
            else:
                G_TAG = 0
            all_stat[model][f"haiku_reward.{task_tag}.K={K}"] = H_TAG
            all_stat[model][f"llama_reward.{task_tag}.K={K}"] = L_TAG
            all_stat[model][f"gpt4t_reward.{task_tag}.K={K}"] = G_TAG
            all_stat[model][f"mixture_of_rewards.{task_tag}.K={K}"] = (H_TAG + L_TAG + G_TAG)/3
           

        all_stat[model][f"haiku_reward.task_macro.K={K}"] = H_TM = haiku_rewards.get(model, {"task_macro_reward": 0})["task_macro_reward"]*100
        all_stat[model][f"llama_reward.task_macro.K={K}"] = L_TM = llama_rewards.get(model, {"task_macro_reward": 0})["task_macro_reward"]*100
        all_stat[model][f"gpt4t_reward.task_macro.K={K}"] = G_TM = gpt4t_rewards.get(model, {"task_macro_reward": 0})["task_macro_reward"]*100

        all_stat[model][f"mixture_of_rewards.K={K}"] = (H + L + G)/3
        all_stat[model][f"task_macro_reward.K={K}"] = (H_TM + L_TM + G_TM)/3


        for task_tag in all_task_types:
            all_stat[model][f"WB_score.{task_tag}"] = scores.get(model, {"task_categorized_scores": {}})["task_categorized_scores"].get(task_tag, 0)*10
        
        all_stat[model][f"WB_score"] = scores.get(model, {"adjusted_score": 0})["adjusted_score"]*10
        all_stat[model][f"WB_score.task_macro"] = scores.get(model, {"adjusted_task_macro_score": 0})["adjusted_task_macro_score"]*10
        all_stat[model][f"Length"] = scores.get(model, {"avg_len": -1})["avg_len"]
         

    for model in all_stat:
        if model not in all_models:
            elo_only_models.append(model)
    # remove the models that are elo only
    for model in elo_only_models:
        del all_stat[model]
    
    # Rank the models by WB_score.task_macro
    pairs_of_modelname_and_score_macro = []
    for model in all_stat:
        pairs_of_modelname_and_score_macro.append((model, all_stat[model]["WB_score.task_macro"]))
    # save the ranks
    pairs_of_modelname_and_score_macro.sort(key=lambda x: x[1], reverse=True)
    for i, (model, score) in enumerate(pairs_of_modelname_and_score_macro):
        all_stat[model]["Rank_ScoreMacro"] = i+1

    pairs_of_modelname_and_task_macro_reward_K = []
    for model in all_stat:
        pairs_of_modelname_and_task_macro_reward_K.append((model, all_stat[model][f"task_macro_reward.K={K}"]))
    # save the ranks
    pairs_of_modelname_and_task_macro_reward_K.sort(key=lambda x: x[1], reverse=True)
    for i, (model, score) in enumerate(pairs_of_modelname_and_task_macro_reward_K):
        all_stat[model][f"Rank_TaskMacroReward.K"] = i+1 
    
    for model in all_stat:
        all_stat[model]["Rank_Avg"] = (all_stat[model]["Rank_ScoreMacro"] + all_stat[model][f"Rank_TaskMacroReward.K"])/2
        all_stat[model]["RewardScore_Avg"] = (all_stat[model]["WB_score.task_macro"] + all_stat[model][f"task_macro_reward.K={K}"])/2
        if model.replace("@together", "") in wb_elo_stat:
            all_stat[model]["WB_Elo"] = wb_elo_stat[model.replace("@together", "")]["avg"]
        else:
            all_stat[model]["WB_Elo"] = "-"
    with open(f"leaderboard/data_dir/all_stat_wildbench.{K}.json", "w") as f:
        json.dump(all_stat, f, indent=2)

    

    # # run python local_scripts/corr_compute.py
    # os.system(f"python local_scripts/corr_compute.py {K}")


for K in [-1, 100, 300, 500, 1000, 1500, 2000, 3000]: 
    merge_scores(K)
    print(f"Finished K={K}")
    # os.system(f"python local_scripts/corr_compute.py {K}")