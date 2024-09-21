import json 
from tabulate import tabulate
import fire 
import os


def show_table(K=-1, mode="main"):
    main_file = f"leaderboard/data_dir/all_stat_wildbench.{K}.json"
    with open(main_file, "r") as f:
        all_stat = json.load(f)

    all_column_names = ['Arena Elo (hard) - 2024-05-20', 'Arena-Hard v0.1', 'AE2.0 LC', 'AE2.0', 'Arena Elo (hard-en) - 2024-06-06', 'haiku_reward.K=$K', 'llama_reward.K=$K', 'gpt4t_reward.K=$K', 'haiku_reward.Creative Tasks.K=$K', 'llama_reward.Creative Tasks.K=$K', 'gpt4t_reward.Creative Tasks.K=$K', 'mixture_of_rewards.Creative Tasks.K=$K', 'haiku_reward.Planning & Reasoning.K=$K', 'llama_reward.Planning & Reasoning.K=$K', 'gpt4t_reward.Planning & Reasoning.K=$K', 'mixture_of_rewards.Planning & Reasoning.K=$K', 'haiku_reward.Math & Data Analysis.K=$K', 'llama_reward.Math & Data Analysis.K=$K', 'gpt4t_reward.Math & Data Analysis.K=$K', 'mixture_of_rewards.Math & Data Analysis.K=$K', 'haiku_reward.Information/Advice seeking.K=$K', 'llama_reward.Information/Advice seeking.K=$K', 'gpt4t_reward.Information/Advice seeking.K=$K', 'mixture_of_rewards.Information/Advice seeking.K=$K', 'haiku_reward.Coding & Debugging.K=$K', 'llama_reward.Coding & Debugging.K=$K', 'gpt4t_reward.Coding & Debugging.K=$K', 'mixture_of_rewards.Coding & Debugging.K=$K', 'haiku_reward.task_macro.K=$K', 'llama_reward.task_macro.K=$K', 'gpt4t_reward.task_macro.K=$K', 'mixture_of_rewards.K=$K', 'task_macro_reward.K=$K', 'WB_score.Creative Tasks', 'WB_score.Planning & Reasoning', 'WB_score.Math & Data Analysis', 'WB_score.Information/Advice seeking', 'WB_score.Coding & Debugging', 'WB_score', 'WB_score.task_macro', 'Length', 'Rank_ScoreMacro', 'Rank_TaskMacroReward.K', 'Rank_Avg', 'RewardScore_Avg', 'WB_Elo']
    all_column_names = [x.replace("$K", str(K)) for x in all_column_names]


    if mode == "main":
        all_column_names_to_show = ["WB_Elo", "RewardScore_Avg", "WB_score.task_macro", f"task_macro_reward.K={K}", "Length"] 
        rank_column = "WB_Elo"
    elif mode == "taskwise_score":
        all_column_names_to_show = ["WB_Elo", "WB_score.task_macro", "WB_score.Creative Tasks", "WB_score.Planning & Reasoning", "WB_score.Math & Data Analysis", "WB_score.Information/Advice seeking", "WB_score.Coding & Debugging", "Length"]
        # rank_column = "WB_score.task_macro"
        rank_column = "WB_Elo"
    elif mode == "taskwise_reward":
        all_column_names_to_show = ["WB_Elo", f"task_macro_reward.K={K}", f"mixture_of_rewards.Creative Tasks.K={K}", f"mixture_of_rewards.Planning & Reasoning.K={K}", f"mixture_of_rewards.Math & Data Analysis.K={K}", f"mixture_of_rewards.Information/Advice seeking.K={K}", f"mixture_of_rewards.Coding & Debugging.K={K}", "Length"]
        rank_column = f"task_macro_reward.K={K}"
    else:
        raise NotImplementedError
    
    # rank by rank_column   
    print(f"Ranking by {rank_column}")
    all_stat = {k: v for k, v in sorted(all_stat.items(), key=lambda item: item[1][rank_column], reverse=True)}
     
    rows = [] 
    for item in all_stat:
        row = [item] + [all_stat[item][x] for x in all_column_names_to_show]
        rows.append(row) 
    
    
    if mode == "taskwise_reward":
        all_column_names_to_show = [x.replace(f".K={K}", "").replace("mixture_of_rewards.","") for x in all_column_names_to_show]
    
    # show a table for the local leaderboard
    # add a rank column 
    print(tabulate(rows, headers=["Model"] + all_column_names_to_show, tablefmt="github", showindex="always", floatfmt=".2f"))


# main 
if __name__ == "__main__":
    fire.Fire(show_table)