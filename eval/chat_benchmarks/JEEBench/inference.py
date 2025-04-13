########################################################################

# from https://github.com/dair-iitd/jeebench/blob/main/inference.py

########################################################################

import argparse
import json
import multiprocessing
import os
from copy import deepcopy
from functools import partial

import openai
from tqdm import tqdm

prompt_library = {
    "MCQ": "In this problem, only one option will be correct. Give a detailed solution and end the solution with the final answer.",
    "MCQ(multiple)": "In this problem, multiple options can be correct. Give a detailed solution and end the solution with the final answer.",
    "Integer": "In this problem, the final answer will be a non-negative integer. Give a detailed solution and end the solution with the final answer.",
    "Numeric": "In this problem, the final will be a numeric value. Give the numerical answer correct upto the 2nd decimal digit. Give a detailed solution and end the solution with the final answer.",
}

few_shot_examples = json.load(open("data/few_shot_examples.json"))


def write_in_file(response_file, response_dict, question, mode, model_nickname):
    if os.path.exists(response_file):
        with open(response_file, "r") as infile:
            responses = json.load(infile)
    else:
        responses = []

    found = False
    for i, old_resp in enumerate(responses):
        if old_resp["description"] == question["description"] and old_resp["index"] == question["index"]:
            responses[i][f"{model_nickname}_{mode}_response"] = response_dict[f"{model_nickname}_{mode}_response"]
            found = True
            break

    if not found:
        responses.append(response_dict)

    json.dump(
        sorted(responses, key=lambda elem: (elem["description"], elem["index"])), open(response_file, "w"), indent=4
    )
    print(f"####UPDATED {response_file}, Current size : {len(responses)}####")


def get_response(question, model, model_nickname, mode, response_file, lock):
    response_dict = deepcopy(question)
    prefix_prompt = prompt_library[question["type"]]
    suffix_prompt = ""

    if mode in ["CoT", "CoT+SC", "CoT+Exam"]:
        suffix_prompt = "Let's think step by step.\n"

    ques = question["question"]
    stripped_ques = ques.replace("\n\n", "\n").strip()
    if mode in ["CoT+OneShot", "CoT", "CoT+SC", "CoT+Exam"]:
        if mode == "CoT+Exam":
            if response_dict["type"] in ["MCQ", "MCQ(multiple)"]:
                if response_dict["type"] == "MCQ":
                    exam_prompt = "If the answer is wrong, you'll be given -1 marks. If the answer is correct, you'll be given +3 marks. If you're unsure of the answer, you can skip the question, and you'll be given 0 marks."
                else:
                    exam_prompt = "If any of the options in the final answer is wrong, you'll be given -2 marks. If all the options are correct, you'll be given +4 marks. If some of the options are correct, you'll be given +1 for each correct option. If you're unsure of the answer, you can skip the question, and you'll be given 0 marks."
                prompt = (
                    prefix_prompt
                    + " "
                    + exam_prompt
                    + "\n\n"
                    + "Problem: "
                    + stripped_ques
                    + "\nSolution: "
                    + suffix_prompt
                )
            else:
                print("No point doing this for Numeric/Integer questions since there is no negative marking...")
                breakpoint()
        else:
            if mode == "CoT+OneShot":
                ex = few_shot_examples[question["subject"]][question["type"]]
                prompt = (
                    prefix_prompt
                    + "\n\n"
                    + "Problem: "
                    + ex["problem"]
                    + "\nSolution: "
                    + ex["solution"]
                    + "\n\n"
                    + "Problem: "
                    + stripped_ques
                    + "\nSolution: "
                )
            else:
                prompt = prefix_prompt + "\n\n" + "Problem: " + stripped_ques + "\nSolution: " + suffix_prompt
    else:
        prompt = prefix_prompt + "\n\n" + "Problem: " + stripped_ques + suffix_prompt
    prompt = prompt.strip()
    response_dict[f"prompt"] = prompt
    num_retries = 0
    print(
        f'Question: {question["description"]}, Index: {question["index"]}, Model: {model_nickname}, Mode: {mode}, query begins'
    )

    while True:
        try:
            if model in ["text-davinci-003", "text-davinci-002", "davinci-002"]:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=2048,
                    temperature=0 if mode in ["CoT", "normal", "CoT+Exam"] else 0.5,
                    n=1 if mode in ["CoT", "normal", "CoT+Exam"] else 3,
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=0 if mode in ["CoT+OneShot", "CoT", "normal", "CoT+Exam"] else 0.5,
                    n=1 if mode in ["CoT+OneShot", "CoT", "normal", "CoT+Exam"] else 8,
                )

            lock.acquire()
            response_dict[f"{model_nickname}_{mode}_response"] = response
            write_in_file(response_file, response_dict, question, mode, model_nickname)
            lock.release()
            break

        except Exception as e:
            num_retries += 1
            print("Failure!", e)
    return


def main():
    """
    The code can restart from the already done questions in case there is a failure midpoint.
    """
    args = argparse.ArgumentParser()
    args.add_argument("--model", default="gpt-3.5-turbo")
    args.add_argument("--data", default="data/dataset.json")
    args.add_argument("--mode", default="normal")
    args.add_argument("--num_procs", default=1, type=int)
    args.add_argument("--max_questions", default=1, type=int)
    args = args.parse_args()

    openai.organization = os.getenv("OPENAI_ORG")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    model_nickname = {
        "davinci-002": "davinci-002",
        "text-davinci-003": "GPT3",
        "gpt-3.5-turbo": "GPT3.5",
        "gpt-4-0613": "GPT4_0613",
        "gpt-4-0314": "GPT4",
    }
    assert args.model in model_nickname.keys()
    assert args.mode in ["normal", "CoT", "CoT+OneShot", "CoT+Exam", "CoT+SC"]

    out_file_dir = f"responses/{model_nickname[args.model]}_{args.mode}_responses"
    out_file = os.path.join(out_file_dir, "responses.json")
    questions = json.load(open(args.data))

    rem_ques = []

    if os.path.exists(out_file):
        for question in tqdm(questions[: args.max_questions]):
            if os.path.exists(out_file):
                with open(out_file, "r") as infile:
                    responses = json.load(infile)
                    found = False

                    for i, old_resp in enumerate(responses):
                        if question["type"] in ["Numeric", "Integer"] and args.mode == "CoT+Exam":
                            found = True
                        if (
                            old_resp["description"] == question["description"]
                            and old_resp["index"] == question["index"]
                        ):
                            found = all(
                                [
                                    old_resp.get(f"{model_nickname[args.model]}_{args.mode}_response", False)
                                    for model in [args.model]
                                ]
                            )
                    if found:
                        print("This question has already been done")
                    else:
                        rem_ques.append(question)
    else:
        os.makedirs(out_file_dir, exist_ok=True)
        if args.mode == "CoT+Exam":
            rem_ques = []
            for q in questions:
                if q["type"] in ["MCQ", "MCQ(multiple)"]:
                    rem_ques.append(q)
        else:
            rem_ques = questions[: args.max_questions]
    print(f"There are {len(rem_ques)} problems remaining")

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pool = multiprocessing.Pool(args.num_procs)
    f = partial(
        get_response,
        model=args.model,
        model_nickname=model_nickname[args.model],
        mode=args.mode,
        response_file=out_file,
        lock=lock,
    )
    pool.map(f, rem_ques)


if __name__ == "__main__":
    main()
