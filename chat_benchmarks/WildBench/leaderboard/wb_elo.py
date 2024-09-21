from leaderboard import data_utils
import json 
import random
from collections import defaultdict
from tqdm import tqdm   
import fire 
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import os
from datetime import datetime

if data_utils.eval_results is None:
    data_utils.load_eval_results()

# eval_results, score_eval_results = load_eval_results()

all_scores_by_id_model = {}
all_outputs_by_id_model = {}
predicted_elos = None 
DATA_DIR = "leaderboard/data_dir/"

def load_scores():
    global all_scores_by_id_model
    if data_utils.score_eval_results is None:
        data_utils.load_eval_results()
    for model_name, model_eval_data in data_utils.score_eval_results.items():
        for item in model_eval_data:
            session_id = item["session_id"]
            score = item["score"]
            if session_id not in all_scores_by_id_model:
                all_scores_by_id_model[session_id] = {}
                all_outputs_by_id_model[session_id] = {}
            all_scores_by_id_model[session_id][model_name] = int(score)
            all_outputs_by_id_model[session_id][model_name] = item["model_output"].strip()
    return 

def get_all_votes(margin=2, tie_margin=1):
    """
    Generate virtual pairwise votes based on the scores.
    """
    global all_scores_by_id_model 
    votes = []
    covered_pairs_ids = set()
    for session_id, scores_by_model in all_scores_by_id_model.items():
        for model_1, score_1 in scores_by_model.items():
            for model_2, score_2 in scores_by_model.items():
                if model_1 == model_2:
                    continue
                vote_item = {}
                vote_item["session_id"] = session_id
                vote_item["model_1"] = model_1
                vote_item["model_2"] = model_2
                vote_item["score_1"] = score_1
                vote_item["score_2"] = score_2
                # decide the empty and truncated 
                model_1_output =  all_outputs_by_id_model[session_id][model_1]
                model_2_output =  all_outputs_by_id_model[session_id][model_2]

                vote_item["length_1"] = len(model_1_output)
                vote_item["length_2"] = len(model_2_output)

                if len(model_1_output) == 0 or len(model_2_output) == 0:
                    continue
                if model_1_output.endswith("... (truncated)") or model_2_output.endswith("... (truncated)"):
                    continue
                if score_1 > score_2 and score_1 - score_2 >= margin:
                    vote_item["winner"] = model_1
                elif score_2 > score_1 and score_2 - score_1 >= margin:
                    vote_item["winner"] = model_2
                else:
                    if abs(score_1 - score_2) <= tie_margin:
                        vote_item["winner"] = "tie"
                    else:
                        continue
                # add to votes after checking if the pair is already covered
                # sort the model ids first 
                model_str = "_".join(sorted([model_1, model_2]))
                uniq_vote_id = f"{session_id}_{model_str}"
                if uniq_vote_id in covered_pairs_ids:
                    continue
                votes.append(vote_item)
                covered_pairs_ids.add(uniq_vote_id)
    return votes

def get_all_votes_from_reward():
    votes = []
    eval_results = data_utils.eval_results
    for eval_mode in data_utils.eval_results:
        for model_name, eval_data in eval_results[eval_mode].items():
            for item in eval_data:
                session_id = item["session_id"]
                result_item = {
                    "session_id": item["session_id"],
                    "model_A": item["model_A"].split("/")[-1],
                    "model_B": item["model_B"].split("/")[-1],
                    "model_A_output": item["model_outputs"][item["model_A"]],
                    "model_B_output": item["model_outputs"][item["model_B"]],
                    "winner": item["winner"],
                    "parsed_result": item["parsed_result"],
                    "extent": item["extent"],   
                }
                if result_item["model_A_output"].endswith("... (truncated)") or result_item["model_B_output"].endswith("... (truncated)"):
                    continue
                if "[This model response is empty.]" in result_item["model_A_output"] or "[This model response is empty.]" in result_item["model_B_output"]:
                    continue
                vote_item = {
                    "session_id": session_id,
                    "model_1": item["model_A"].split("/")[-1],
                    "model_2": item["model_B"].split("/")[-1],
                    "winner": item["winner"],
                }
                if result_item["extent"] == 2:
                    votes.append(vote_item)
                else:
                # elif result_item["extent"] == 0:
                    vote_item["winner"] = "tie"
                    votes.append(vote_item)
    return votes 
 
def compute_single_round(votes, K, init_elos, dynamic, interval=10, use_regressed_as_init=False, length_margin=-1):
    elo = init_elos.copy() if init_elos is not None else {}
    # load predicted elo as init for other models.
    if use_regressed_as_init:
        predicted_elos = load_predicted_elo()
        for model in predicted_elos:
            # if model not in elo:
            elo[model] = predicted_elos[model]
    # sample_votes = [random.choice(votes) for _ in range(len(votes))]
    # shuffle the votes
    sample_votes = random.sample(votes, len(votes))

    # Initialize Elo ratings
    for vote in sample_votes:
        if vote["model_1"] not in elo:
            elo[vote["model_1"]] = 1000
        if vote["model_2"] not in elo:
            elo[vote["model_2"]] = 1000

    vote_update_cnt = defaultdict(int)
    # Calculate Elo ratings for the bootstrap sample
    for vote in sample_votes:
        model_1 = vote["model_1"]
        model_2 = vote["model_2"]
        if model_1 in init_elos and model_2 in init_elos:
            continue

        elo_1 = elo[model_1]
        elo_2 = elo[model_2]

        expected_1 = 1 / (1 + 10 ** ((elo_2 - elo_1) / 400))
        expected_2 = 1 / (1 + 10 ** ((elo_1 - elo_2) / 400))

        if length_margin > 0:
            abs_len_diff = abs(vote["length_1"] - vote["length_2"])
            if abs_len_diff > length_margin:
                vote["winner"] = "tie"

        if vote["winner"] == model_1:
            score_1 = 1
            score_2 = 0
        elif vote["winner"] == model_2:
            score_1 = 0
            score_2 = 1
        else:
            score_1 = 0.5
            score_2 = 0.5

        if model_1 not in init_elos:
            elo[model_1] += K * (score_1 - expected_1)
        else:
            if dynamic:
                elo[model_1] += K * (score_1 - expected_1)
                if interval > 0 and vote_update_cnt[model_1] % interval == 0:
                    elo[model_1] = (elo[model_1] + init_elos[model_1]) / 2

        if model_2 not in init_elos:
            elo[model_2] += K * (score_2 - expected_2)
        else:
            if dynamic:
                elo[model_2] += K * (score_2 - expected_2)
                if interval > 0 and vote_update_cnt[model_2] % interval == 0:
                    elo[model_2] = (elo[model_2] + init_elos[model_2]) / 2
    
        vote_update_cnt[model_1] += 1
        vote_update_cnt[model_2] += 1

    return elo

def compute_elo_based_on_votes(votes, K=4, num_rounds=1000, init_elos=None, dynamic=False, num_processes=None, interval=10, use_regressed_as_init=False, length_margin=-1):
    """
    Compute Elo rating based on votes with bootstrapping method using multiprocessing.
    """
    elo_cumulative = defaultdict(list)
    num_models = defaultdict(int)

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(compute_single_round, votes, K, init_elos, dynamic, interval, use_regressed_as_init, length_margin) for _ in range(num_rounds)]
        for future in tqdm(as_completed(futures), total=num_rounds):
            elo = future.result()
            for model, rating in elo.items():
                elo_cumulative[model].append(rating)
                num_models[model] += 1

    elo_avg = {model: sum(ratings) / num_models[model] for model, ratings in elo_cumulative.items()}
    elo_std = {model: (sum((rating - elo_avg[model]) ** 2 for rating in ratings) / num_models[model]) ** 0.5 for model, ratings in elo_cumulative.items()}
    elo_ci_lower = {}
    elo_ci_upper = {}
    for model, ratings in elo_cumulative.items():
        ci_lower = np.percentile(ratings, 2.5)
        ci_upper = np.percentile(ratings, 97.5)
        elo_ci_lower[model] = ci_lower
        elo_ci_upper[model] = ci_upper

    elo_ci = {model: (elo_ci_lower[model], elo_ci_upper[model]) for model in elo_avg.keys()}
    elo_median = {model: np.median(ratings) for model, ratings in elo_cumulative.items()}
    return elo_avg, elo_std, elo_median, elo_ci

def load_init_elo(filepath = DATA_DIR+ "all_stat.json", elo_key = "Arena Elo (hard-en) - latest"):
    init_elos = {} 
    with open(filepath, "r") as f:
        data = json.load(f)
        for model in data:
            model = model.replace("@together", "")
            elo = data[model].get(elo_key, "-")
            if elo != "-":
                init_elos[model] = float(elo)
    print(f">>> Loaded {len(init_elos)} init elos with the key {elo_key}")
    return init_elos

def load_predicted_elo(filepath = DATA_DIR+ "wb_elo_regression.json", elo_key = "Predicted Elo"):
    global predicted_elos 
    if predicted_elos is None:
        predicted_elos = {}
        with open(filepath, "r") as f:
            data = json.load(f)
            for model in data:
                # model = model.replace("@together", "")
                elo = data[model].get(elo_key, "-")
                if elo != "-":
                    model = model.replace("@together", "")
                    predicted_elos[model] = float(elo)
        print(f">>> Loaded {len(predicted_elos)} predicted elos with the key {elo_key}")
    return predicted_elos

def compute_wb_elo(loo=-1, seed=42, margin=2, K=4, num_rounds=10, tie_margin=1, dynamic=False, num_processes=1, interval=10, use_regressed_as_init=False, length_margin=-1): 
    global all_scores_by_id_model

    random.seed(seed) 
    init_elos = load_init_elo() 


    if all_scores_by_id_model == {}:
        load_scores()
    
    
    print(f">>> Config: WB Elo with K={K} and num_rounds={num_rounds}; margin={margin}; loo={loo}; seed={seed}; init_elo={len(init_elos)} models; tie_margin={tie_margin}; dynamic={dynamic}; num_processes={num_processes}; interval={interval}; use_regressed_as_init={use_regressed_as_init}; length_margin={length_margin}")

    if loo >= 0 and loo < len(init_elos):    
        ranked_init_elos = {k: v for k, v in sorted(init_elos.items(), key=lambda item: item[1], reverse=True)} 
        # print(json.dumps(ranked_init_elos, indent=4))
        # LEAVE ONE OUT for cross-validation 
        random_selected_model = list(ranked_init_elos.keys())[loo]
        print(f">>> Randomly selected model to remove from init_elo : {random_selected_model}")
        elo_for_random_selected_model = init_elos[random_selected_model]
        init_elos.pop(random_selected_model)
        # get a random key in all_scores_by_id_model
        sid = random.choice(list(all_scores_by_id_model.keys()))
        if random_selected_model not in all_scores_by_id_model[sid]:
            print(f">>> Model {random_selected_model} not in the scores")
            return
    elif loo >= len(init_elos):
        print(f">>> LOO index {loo} is out of range")
        return 
    
    votes = get_all_votes(margin, tie_margin)
    # votes += get_all_votes_from_reward()

    print(f">>> Found {len(votes)} votes")
    # non-tie votes
    non_tie_votes = [item for item in votes if item["winner"] != "tie"]
    print(f">>> Found {len(non_tie_votes)} non-tie votes")
 
    elo_avg, elo_std, elo_median, elo_ci = compute_elo_based_on_votes(votes, K=K, num_rounds=num_rounds, init_elos=init_elos, dynamic=dynamic, num_processes=num_processes, interval=interval, use_regressed_as_init=use_regressed_as_init, length_margin=length_margin)
    # rank by elo
    elo_stat = {k: {"avg": v, "std": elo_std[k], \
                     "median": elo_median[k], "ci": elo_ci[k],
                      "init_elo": init_elos.get(k, '-')} \
                for k, v in sorted(elo_avg.items(), key=lambda item: item[1], reverse=True)}
    print(f">>> WB Elo with K={K} and num_rounds={num_rounds}")
    # print(json.dumps(elo_stat, indent=4))

    if loo > -1 and random_selected_model in elo_avg: 
        estimated_elo_for_random_selected_model = elo_avg[random_selected_model]
        print(f">>> Init Elo for {random_selected_model} (hidden) : {elo_for_random_selected_model}")
        print(f">>> Estimated Elo for {random_selected_model} : {estimated_elo_for_random_selected_model}")
        diff = elo_for_random_selected_model - estimated_elo_for_random_selected_model
        print(f">>> Diff for {random_selected_model} : {diff}")

    elo_results = {
        "config": {
            "K": K,
            "num_rounds": num_rounds,
            "margin": margin,
            "tie_margin": tie_margin,
            "dynamic": dynamic,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "interval": interval,
            "use_regressed_as_init": use_regressed_as_init,
            "length_margin": length_margin,
        },
        "elo_stat": elo_stat
    }
    with open(f"{DATA_DIR}/wb_elo_results.json", "w") as f:
        json.dump(elo_results, f, indent=4)
        print(f">>> Saved WB Elo results to {f.name}")

if __name__ == "__main__":
    fire.Fire(compute_wb_elo)
 

"""
margin=3;tie_margin=2;K=4;dynamic=True;interval=16; LM=-1
python -m leaderboard.wb_elo --K $K --margin $margin --tie_margin $tie_margin --num_rounds 100 --dynamic $dynamic --interval $interval --num_processes 4 --length_margin $LM
"""