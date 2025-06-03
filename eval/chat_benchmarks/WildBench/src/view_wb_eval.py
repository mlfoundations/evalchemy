import os
import json
from tabulate import tabulate
import sys

#### task tag mapping ####
from datasets import load_dataset

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

print(list(set(task_group_new.values())))

task_mapping = {}
wb_data = load_dataset("allenai/WildBench", "v2", split="test")
for item in wb_data:
    tags = [item["primary_tag"]] + item["secondary_tags"]
    task_mapping[item["id"]] = []
    for tag in tags:
        task_mapping[item["id"]].append(task_group_new[tag])


FOLDER = "eval_results/v2.0522"
ACTION = sys.argv[1]
K = -1  # for pairwise length margin
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
elif ACTION == "score.legacy":
    folder = FOLDER + "/score.v2/eval=gpt-4-turbo-2024-04-09/"
    MODE = "score"
elif ACTION == "score":
    folder = FOLDER + "/score.v2/eval=gpt-4o-2024-05-13/"
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
for file in files:
    if file.endswith(".json"):
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
                extent = item["extent"]
                winner = item["winner"]
                if winner == test_model_id:
                    if extent == 2:
                        win_much_counts.append(1)
                    elif extent == 1:
                        if K >= 0 and test_model_output_len > ref_model_output_len + K:
                            tie_counts.append(1)
                        else:
                            win_counts.append(1)
                elif winner == ref_model_id:
                    if extent == 2:
                        lose_much_counts.append(1)
                    elif extent == 1:
                        if K >= 0 and ref_model_output_len > test_model_output_len + K:
                            tie_counts.append(1)
                        else:
                            lose_counts.append(1)
                elif winner == "tie":
                    tie_counts.append(1)

                lengths.append(test_model_output_len)

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
            }
            row_item["reward"] = (
                row_item["win"] * 0.5
                + row_item["win_much"] * 1
                + row_item["tie"] * 0
                - row_item["lose"] * 0.5
                - row_item["lose_much"] * 1
            )
            row_item["reward"] = row_item["reward"] / row_item["total"]
            row_item["K"] = K
            # row_item["win_rate"] = (row_item["win"] + row_item["win_much"]) / row_item["total"]
        elif MODE == "score":
            task_cat_results = {}
            for item in eval_result:
                scores.append(float(item["score"]))
                model_output = item["model_output"].strip()
                model_output_len = len(model_output)
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
            task_macro_score = sum(task_cat_score.values()) / len(task_cat_score)

            row_item = {
                "model": file.replace(".json", ""),
                "raw_score": sum(scores) / len(scores),
                "adjusted_score": (sum(scores) / len(scores) - 5) * 2 * 10,
                "task_macro_score": task_macro_score * 10,
                "total": len(eval_result),
                "avg_len": sum(lengths) / len(lengths),
            }

            for tag in task_cat_score:
                row_item[tag] = task_cat_score[tag] * 10

        table.append(row_item)
if MODE == "pairwise":
    table = sorted(table, key=lambda x: x["reward"] * 100, reverse=True)
elif MODE == "score":
    table = sorted(table, key=lambda x: x["task_macro_score"], reverse=True)

result = {}
for item in table:
    name = item["model"]
    if "/" in name:
        name = name.split("/")[-1]
    result[name] = item

if MODE == "pairwise":
    ACTION = f"{ACTION}-K={K}"

print(tabulate(table, headers="keys", tablefmt="grid"))

# with open(f"data_dir/{ACTION}.json", "w") as f:
#     json.dump(result, f, indent=2)
