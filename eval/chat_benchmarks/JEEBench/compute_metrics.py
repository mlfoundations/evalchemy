########################################################################

# from https://github.com/dair-iitd/jeebench/blob/main/compute_metrics.py

########################################################################

import json

import numpy as np
import pandas as pd

QUES_TYPES = ["MCQ", "MCQ(multiple)", "Integer", "Numeric"]

models = [
    "Random",
    "GPT3_normal",
    "GPT3.5_normal",
    "GPT4_normal",
    "GPT4_CoT",
    "GPT4_CoT_self_refine",
    "GPT4_CoT+OneShot",
    "GPT4_CoT+SC@8",
]


def get_aggregate(answers, question_type, single_threshold=None, multiple_threshold=None):
    # Pass optional \tau_{single} and \tau_{multiple} parameters if needed for evaluation under risk.
    if question_type == "MCQ(multiple)" or question_type == "MCQ":
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "None": 4}
        idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "None"}
        abcd = [0, 0, 0, 0, 0]
        for ans in answers:
            if ans == "None":
                abcd[letter_to_idx[ans]] += 1
            else:
                for c in ans:
                    abcd[letter_to_idx[c]] += 1
        if question_type == "MCQ":
            abcd = abcd[:-1]
            answer = idx_to_letter[np.argmax(abcd)]
            if single_threshold is not None:
                answer = answer if abcd[np.argmax(abcd)] / len(answers) >= single_threshold else "None"
        else:
            if multiple_threshold is not None:
                options_selected = [
                    idx_to_letter[x]
                    for x in range(len(abcd))
                    if abcd[x] >= len(answers) * multiple_threshold and idx_to_letter[x] != "None"
                ]
            else:
                options_selected = [
                    idx_to_letter[x]
                    for x in range(len(abcd))
                    if abcd[x] >= len(answers) / 2 and idx_to_letter[x] != "None"
                ]
            if len(options_selected) == 0:
                answer = "None"
            else:
                answer = "".join(sorted(options_selected))
    else:  # For integer and numeric answers, choose the most common response(other than None)
        while "None" in answers:
            answers.remove("None")
        if len(answers) == 0:
            answers = ["None"]
        unique, counts = np.unique(answers, return_counts=True)
        answer = unique[np.argmax(counts)]
    return answer


def compute_score(gold, resp, question_type, year):
    assert question_type in QUES_TYPES
    if question_type == "MCQ(multiple)":
        gold = set([c for c in ["A", "B", "C", "D"] if c in gold])
        resp = set([c for c in ["A", "B", "C", "D"] if c in resp])
        if resp == gold:
            return 1.0
        else:
            if len(resp - gold) == 0:
                return 0.25 * len(resp)
            return 0.0  # If response contains something not in the gold set, give 0
    elif question_type == "MCQ":
        gold = set([c for c in ["A", "B", "C", "D"] if c in gold])
        resp = set([c for c in ["A", "B", "C", "D"] if c in resp])
        return int(gold == resp)
    else:
        if resp == "None":
            return 0.0
        g, r = float(gold), float(resp)
        return int(abs(g - r) <= 0.01)


def construct_responses_table():
    responses = {}
    for model in models:
        if "SC@" in model:
            pass
        elif "Random" == model:
            pass
        else:
            responses[model] = json.load(open(f"data/responses/{model}_responses/responses.json"))
    dataset = json.load(open("data/dataset.json"))
    extracts = {
        "Type": [],
        "Index": [],
        "Description": [],
        "Subject": [],
        "Gold": [],
    }
    for model in models:
        if "Random" == model:
            continue
        else:
            extracts[f"{model}"] = []

    for i, q in enumerate(dataset):
        extracts["Type"].append(q["type"])
        extracts["Index"].append(q["index"])
        extracts["Description"].append(q["description"])
        extracts["Subject"].append(q["subject"])
        extracts["Gold"].append(q["gold"])

        for model in models:
            if "SC@" in model:
                continue
            elif "Random" == model:
                continue
            else:
                try:
                    assert q["question"] == responses[model][i]["question"]
                except:
                    print(q["question"])
                    breakpoint()
                    print(responses[model][i]["question"])
                    breakpoint()
                try:
                    extracts[f"{model}"].append(responses[model][i]["extract"])
                except:
                    print(extracts)

    if "GPT4_CoT+SC" in model:
        num_responses = int(model.split("@")[1])
        for i, q in enumerate(dataset):
            sc_responses = json.load(open("data/responses/GPT4_CoT+SC_responses/responses.json"))
            resp = sc_responses[i]
            answers = [resp["GPT4_CoT+SC_response"]["choices"][k]["extract"] for k in range(num_responses)]
            answer = get_aggregate(answers, resp["type"])

            extracts[f"{model}"].append(answer)
    pd.DataFrame(extracts).to_csv("results/extracts.csv", index=False)

    return pd.read_csv("results/extracts.csv", dtype=str)


if __name__ == "__main__":
    responses = construct_responses_table()
    output = []
    for i, response in responses.iterrows():
        out = {}
        out["Type"] = response["Type"]
        out["Index"] = response["Index"]
        out["Description"] = response["Description"]
        out["Subject"] = response["Subject"]
        gold = response["Gold"]
        out["Gold"] = gold
        if response["Type"] == "MCQ":
            out["Random"] = 0.25
        elif response["Type"] == "MCQ(multiple)":
            num_ans = len(gold)
            if num_ans == 1:
                out["Random"] = 0.0625
            elif num_ans == 2:
                out["Random"] = 0.09375
            elif num_ans == 3:
                out["Random"] = 0.203125
            elif num_ans == 4:
                out["Random"] = 0.5
        else:
            out["Random"] = 0

        for model in models:
            if model == "Random":
                continue
            resp = response[f"{model}"]
            if not isinstance(resp, str):
                resp = "None"
            out[f"{model}"] = resp
            out[f"{model}"] = compute_score(gold, resp, out["Type"], out["Description"])
        out[f"Max"] = 1
        output.append(out)

    df = pd.DataFrame()
    df["Type"] = [x["Type"] for x in output]
    df["Index"] = [x["Index"] for x in output]
    df["Description"] = [x["Description"] for x in output]
    df["Subject"] = [x["Subject"] for x in output]
    df["Gold"] = [x["Gold"] for x in output]
    df["Random"] = [x["Random"] for x in output]
    for model in models:
        df[f"{model}"] = [x.get(f"{model}", "None") for x in output]
        df[f"{model}"] = [x.get(f"{model}", 0) for x in output]

    df.to_csv(f"results/scores.csv", index=False)

    modes = ["overall", "type_wise", "subject_wise"]
    for mode in modes:
        col_dict = {}
        for model in models:
            col_dict[f"{model}"] = ["mean"]

        if mode != "overall":
            col_dict[f"{models[0]}"].insert(0, "count")

        if mode == "overall":
            grouped_multiple = df.agg(col_dict)
        elif mode == "type_wise":
            grouped_multiple = df.groupby(["Type"]).agg(col_dict)
        elif mode == "subject_wise":
            grouped_multiple = df.groupby(["Subject"]).agg(col_dict)

        if mode != "overall":
            grouped_multiple.columns = ["count"] + models
        grouped_multiple = grouped_multiple.reset_index()
        grouped_multiple = grouped_multiple.round(3)
        grouped_multiple.to_csv(f"results/aggregated_scores_{mode}.csv", index=False)
    print("Done!")
