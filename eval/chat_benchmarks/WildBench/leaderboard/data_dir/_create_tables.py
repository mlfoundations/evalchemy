import os
import json
from tabulate import tabulate
import sys
from datasets import load_dataset
from tqdm import tqdm

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
    "Others": "Creative Tasks",
}

# print(list(set(task_group_new.values())))

task_mapping = {}
wb_data = load_dataset("allenai/WildBench", "v2", split="test")
for item in wb_data:
    tags = [item["primary_tag"]] + item["secondary_tags"]
    task_mapping[item["id"]] = []
    for tag in tags:
        task_mapping[item["id"]].append(task_group_new[tag])

    # deduplicate
    task_mapping[item["id"]] = list(set(task_mapping[item["id"]]))

    # # remove "Others"
    # if "Others" in task_mapping[item["id"]]:
    #     task_mapping[item["id"]].remove("Others")

# all_task_types = ['Information seeking', 'Creative Writing', 'Coding & Debugging', 'Reasoning', 'Editing', 'Math', 'Planning', 'Brainstorming', 'Role playing', 'Advice seeking', 'Data Analysis']

PAIRWISE_FOLDER = "eval_results/v2.0522"
SCORE_FOLDER = "eval_results/v2.0625"
ACTION = sys.argv[1]
K = -1  # for pairwise length margin

if ACTION.startswith("pairwise"):
    FOLDER = PAIRWISE_FOLDER
elif ACTION.startswith("score"):
    FOLDER = SCORE_FOLDER

if ACTION == "pairwise-gpt4t":
    folder = FOLDER + "/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=gpt-4-turbo-2024-04-09"
    MODE = "pairwise"
    ref_model = "gpt-4-turbo-2024-04-09"
elif ACTION == "pairwise-llama":
    folder = FOLDER + "/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=Llama-2-70b-chat-hf"
    MODE = "pairwise"
    ref_model = "Llama-2-70b-chat-hf"
elif ACTION == "pairwise-haiku":
    folder = FOLDER + "/pairwise.v2/eval=gpt-4-turbo-2024-04-09/ref=claude-3-haiku-20240307"
    MODE = "pairwise"
    ref_model = "claude-3-haiku-20240307"
elif ACTION == "score":
    # folder = FOLDER+"/score.v2/eval=gpt-4-turbo-2024-04-09/"
    folder = FOLDER + "/score.v2/eval=gpt-4o-2024-05-13/"
    MODE = "score"
elif ACTION == "score-sonnet":
    folder = FOLDER + "/score.v2/eval=claude-3-5-sonnet-20240620/"
    MODE = "score"
else:
    print("Please provide either 'pairwise' or 'score' as the argument")
    sys.exit()

if MODE == "pairwise":
    if len(sys.argv) == 3:
        K = int(sys.argv[2])
        print(f"Using K={K} as the margin for pairwise comparison")

# list all files
files = os.listdir(folder)
table = []
for file in tqdm(files, desc=f"Processing {folder.replace(FOLDER, '')}"):
    if file.endswith(".json") and not any(
        [x in file for x in ["128", "256", "384", "512", "640", "768", "896", "1024"]]
    ):
        print(f"Processing {file}")
        eval_result = []
        with open(f"{folder}/{file}", "r") as f:
            eval_result = json.load(f)
        win_much_counts = []
        win_counts = []
        tie_counts = []
        lose_counts = []
        lose_much_counts = []
        lengths = []
        scores = []
        task_cat_results = {}
        if MODE == "pairwise":
            model_lists = list(eval_result[0]["model_outputs"].keys())
            if len(model_lists) == 1:
                # ref_model_id = model_lists[0]
                # test_model_id = model_lists[0]
                continue
            else:
                ref_model_id = model_lists[0] if ref_model in model_lists[0] else model_lists[1]
                test_model_id = model_lists[0] if ref_model in model_lists[1] else model_lists[1]

            for item in eval_result:
                test_model_output_len = len(item["model_outputs"][test_model_id])
                ref_model_output_len = len(item["model_outputs"][ref_model_id])

                test_model_truncated = item["model_outputs"][test_model_id].endswith("... (truncated)")
                ref_model_truncated = item["model_outputs"][ref_model_id].endswith("... (truncated)")
                test_model_empty = item["model_outputs"][test_model_id] == "[This model response is empty.]"
                ref_model_empty = item["model_outputs"][ref_model_id] == "[This model response is empty.]"

                if test_model_truncated or ref_model_truncated:
                    continue
                if test_model_empty or ref_model_empty:
                    continue

                extent = item["extent"]
                winner = item["winner"]
                result_label = None

                task_tags = task_mapping[item["session_id"]]

                if winner == test_model_id:
                    if extent == 2:
                        win_much_counts.append(1)
                        result_label = "win_much"
                    elif extent == 1:
                        if K >= 0 and test_model_output_len > ref_model_output_len + K:
                            tie_counts.append(1)
                            result_label = "tie"
                        else:
                            win_counts.append(1)
                            result_label = "win"
                elif winner == ref_model_id:
                    if extent == 2:
                        lose_much_counts.append(1)
                        result_label = "lose_much"
                    elif extent == 1:
                        if K >= 0 and ref_model_output_len > test_model_output_len + K:
                            tie_counts.append(1)
                            result_label = "tie"
                        else:
                            lose_counts.append(1)
                            result_label = "lose"
                elif winner == "tie":
                    tie_counts.append(1)
                    result_label = "tie"

                assert result_label is not None

                lengths.append(test_model_output_len)

                # For task-based analysis

                for tag in task_tags:
                    if tag not in task_cat_results:
                        task_cat_results[tag] = {"win_much": 0, "win": 0, "tie": 0, "lose": 0, "lose_much": 0}
                    task_cat_results[tag][result_label] += 1

            task_cat_reward = {}  # compute the rewards for each task category
            for tag in task_cat_results:
                item = task_cat_results[tag]
                task_instance_num = sum(item.values())
                reward = (
                    item["win"] * 0.5
                    + item["win_much"] * 1
                    + item["tie"] * 0
                    - item["lose"] * 0.5
                    - item["lose_much"] * 1
                )
                # try:
                reward = reward / task_instance_num
                # except ZeroDivisionError:
                #     print(tag)
                #     print(item)
                #     exit()
                task_cat_reward[tag] = reward

            row_item = {
                # "model": test_model_id,
                "model": file.replace(".json", ""),
                "win_much": sum(win_much_counts),
                "win": sum(win_counts),
                "tie": sum(tie_counts),
                "lose": sum(lose_counts),
                "lose_much": sum(lose_much_counts),
                # "avg_score": sum(scores) / len(scores),
                "total": len(eval_result),
                "avg_len": sum(lengths) / len(lengths),
                "task_categorized_results": task_cat_results,
                "task_categorized_rewards": task_cat_reward,
            }
            row_item["reward"] = (
                row_item["win"] * 0.5
                + row_item["win_much"] * 1
                + row_item["tie"] * 0
                - row_item["lose"] * 0.5
                - row_item["lose_much"] * 1
            )
            row_item["reward"] = row_item["reward"] / row_item["total"]
            weights_by_task = {
                "Creative Tasks": 0.5,
                "Planning & Reasoning": 1.25,
                "Math & Data Analysis": 1,
                "Information/Advice seeking": 0.75,
                "Coding & Debugging": 1.25,
            }
            # row_item["task_macro_reward"] = sum(task_cat_reward.values()) / len(task_cat_reward)
            row_item["task_macro_reward"] = sum(
                [task_cat_reward[tag] * weights_by_task[tag] for tag in task_cat_reward]
            ) / sum(weights_by_task.values())
            row_item["K"] = K
            # row_item["win_rate"] = (row_item["win"] + row_item["win_much"]) / row_item["total"]
        elif MODE == "score":
            task_cat_results = {}
            for item in eval_result:
                # print(item.keys())
                if ACTION == "score-sonnet" and "parsed_result" in item:
                    item["score"] = item["parsed_result"]["score"]
                    if type(item["model_output"]) == list:
                        item["model_output"] = item["model_output"][0]
                    item["model_test"] = item["generator"]
                if "score" not in item:
                    print(item)
                scores.append(float(item["score"]))
                model_output = item["model_output"]
                if model_output.endswith("... (truncated)"):
                    continue
                model_output_len = len(model_output)
                if model_output_len == 0:
                    continue
                lengths.append(model_output_len)
                task_tags = task_mapping[item["session_id"]]
                for tag in task_tags:
                    if tag not in task_cat_results:
                        task_cat_results[tag] = []
                    task_cat_results[tag].append(float(item["score"]))
            test_model_id = item["model_test"]
            task_cat_score = {}
            for tag in task_cat_results:
                task_cat_score[tag] = sum(task_cat_results[tag]) / len(task_cat_results[tag])
                # adjust
                task_cat_score[tag] = (task_cat_score[tag] - 5) * 2
            weights_by_task = {
                "Creative Tasks": 0.5,
                "Planning & Reasoning": 1.25,
                "Math & Data Analysis": 1,
                "Information/Advice seeking": 0.75,
                "Coding & Debugging": 1.25,
            }
            # task_macro_score = sum(task_cat_score.values()) / len(task_cat_score)
            task_macro_score = sum([task_cat_score[tag] * weights_by_task[tag] for tag in task_cat_score]) / sum(
                weights_by_task.values()
            )
            row_item = {
                "model": file.replace(".json", ""),
                "score": sum(scores) / len(scores),
                "adjusted_score": (sum(scores) / len(scores) - 5) * 2,
                "task_macro_score": task_macro_score,
                "adjusted_task_macro_score": task_macro_score,
                "task_categorized_scores": task_cat_score,
                "total": len(eval_result),
                "avg_len": sum(lengths) / len(lengths),
            }
        table.append(row_item)
if MODE == "pairwise":
    table = sorted(table, key=lambda x: x["reward"], reverse=True)
elif MODE == "score":
    table = sorted(table, key=lambda x: x["score"], reverse=True)
# print the table with grid format and .2f for float numbers
# print(tabulate(table, headers="keys", tablefmt="grid", floatfmt=".2f"))

# save tsv file to a local file
# with open(f"local_scripts/{ACTION}.tsv", "w") as f:
#     f.write(tabulate(table, headers="keys", tablefmt="tsv", floatfmt=".2f"))

# write a json file where each key is the model name
result = {}
for item in table:
    name = item["model"]
    if "/" in name:
        name = name.split("/")[-1]
    result[name] = item

if MODE == "pairwise":
    ACTION = f"{ACTION}-K={K}"

with open(f"leaderboard/data_dir/{ACTION}.json", "w") as f:
    json.dump(result, f, indent=2)

"""
python data_dir/_create_tables.py score-sonnet
python data_dir/_create_tables.py score
python data_dir/_create_tables.py pairwise-gpt4t -1
python data_dir/_create_tables.py pairwise-llama -1
python data_dir/_create_tables.py pairwise-haiku -1

python data_dir/_create_tables.py pairwise-gpt4t 100
python data_dir/_create_tables.py pairwise-llama 100
python data_dir/_create_tables.py pairwise-haiku 100

python data_dir/_create_tables.py pairwise-gpt4t 300
python data_dir/_create_tables.py pairwise-llama 300
python data_dir/_create_tables.py pairwise-haiku 300

python data_dir/_create_tables.py pairwise-gpt4t 500
python data_dir/_create_tables.py pairwise-llama 500
python data_dir/_create_tables.py pairwise-haiku 500

python data_dir/_create_tables.py pairwise-gpt4t 1000
python data_dir/_create_tables.py pairwise-llama 1000
python data_dir/_create_tables.py pairwise-haiku 1000

# python data_dir/_create_tables.py pairwise-gpt4t 3000
# python data_dir/_create_tables.py pairwise-llama 3000
# python data_dir/_create_tables.py pairwise-haiku 3000

# python data_dir/_create_tables.py pairwise-gpt4t 10000
# python data_dir/_create_tables.py pairwise-llama 10000
# python data_dir/_create_tables.py pairwise-haiku 10000
"""
