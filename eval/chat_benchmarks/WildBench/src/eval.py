import argparse
import os
import json
import openai
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor

# from eval_utils import (
# retry_handler,
# openai_chat_request,
# anthropic_chat_request,
# )

from .unified_utils import (
    retry_handler,
    openai_chat_request,
    anthropic_chat_request,
)
from datasets import load_dataset, get_dataset_config_names
import tiktoken
import re

HF_BENCH_PATH = "allenai/WildBench"
HF_BENCH_CONFIG = "v2"
HF_RESULTS_PATH = "allenai/WildBench-V2-Model-Outputs"


print(f"Loading the benchmark data from {HF_BENCH_PATH} and the results from {HF_RESULTS_PATH}")

encoding = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="trial", required=True)
    parser.add_argument("--mode", type=str, default="pairwise", required=True)
    parser.add_argument("--eval_template", type=str, default="", required=True)
    parser.add_argument("--target_model_name", type=str, required=False)
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--ref_model_name", type=str, required=False)
    parser.add_argument("--eval_output_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=1)

    # Prompt configs
    parser.add_argument("--max_words_to_eval", type=int, default=1000)

    # OpenAI Configs
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_mode", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_result_file", type=str, default=None)

    args = parser.parse_args()
    if args.api_key is not None:
        openai.api_key = args.api_key

    return args


def extract_values_from_json(json_string, keys=["score", "strengths", "weaknesses", "choice"], allow_no_quotes=False):
    extracted_values = {}
    for key in keys:
        if key not in json_string:
            continue
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f"{key}\\s*:\\s*([^,\\s]*)"
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


def parse_result(result_str, mode="json", eval_mode="score"):
    assert eval_mode in ["score", "pairwise"]
    result_str = result_str.strip()
    try:
        # result_str = result_str.replace(".\n", ". ")
        # result_str = result_str.replace(".\n\n", ". ")
        # result_str = result_str.replace("\n", " ")
        try:
            parsed_result = json.loads(result_str)
        except:
            parsed_result = extract_values_from_json(result_str, keys=["score", "choice"])
    except Exception as e:
        print(result_str)
        print(e)
        # raise Exception(f"Failed to parse the result: {result_str}")
        parsed_result = {"N/A": "N/A"}
        # exit()
    return parsed_result


def compute_cost(gpt_model_name, prompt, result):
    global encoding
    if encoding is None:
        if gpt_model_name.startswith("gpt"):
            encoding = tiktoken.encoding_for_model(gpt_model_name)
        else:
            encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")  # default to gpt-4-1106-preview
    if gpt_model_name.startswith("gpt-4-"):  # gpt-4-turbo series
        price_per_input_token = 0.01 / 1000
        price_per_output_token = 0.03 / 1000
    elif gpt_model_name.startswith("gpt-4o"):  # gpt-4-turbo series
        price_per_input_token = 0.005 / 1000
        price_per_output_token = 0.015 / 1000
    elif gpt_model_name in ["gpt-3.5-turbo-0125"]:
        price_per_input_token = 0.0005 / 1000
        price_per_output_token = 0.0015 / 1000
    elif gpt_model_name in ["gpt-4"]:
        price_per_input_token = 0.03 / 1000
        price_per_output_token = 0.06 / 1000
    elif gpt_model_name.startswith("claude-3-opus"):
        price_per_input_token = 0.015 / 1000
        price_per_output_token = 0.075 / 1000
    elif gpt_model_name.startswith("claude-3-sonnet"):
        price_per_input_token = 0.003 / 1000
        price_per_output_token = 0.015 / 1000
    elif gpt_model_name.startswith("claude-3-haiku"):
        price_per_input_token = 0.00025 / 1000
        price_per_output_token = 0.00125 / 1000
    else:
        # raise Exception(f"Unknown model: {gpt_model_name}")
        # print(f"Unknown model: {gpt_model_name}")
        pass
        price_per_input_token = 0.0
        price_per_output_token = 0.0

    price_item = {
        # compute openai token number
        "in_tokens": len(encoding.encode(prompt)),
        "out_tokens": len(encoding.encode(result)),
    }
    price_item["cost"] = (
        price_item["in_tokens"] * price_per_input_token + price_item["out_tokens"] * price_per_output_token
    )
    return price_item


def batch_eval_generate(results, args):
    json_lines = []
    for ind, item in tqdm(enumerate(results), total=len(results)):
        sid = item["session_id"]
        batch_item = {}

        if args.mode == "pairwise":
            model_A = item["assignment"]["A"]
            model_B = item["assignment"]["B"]
            batch_item["custom_id"] = f"{sid}||A:{model_A}||B:{model_B}"
        elif args.mode == "score":
            model_test = item["generator"]
            batch_item["custom_id"] = f"{sid}||{model_test}"

        batch_item["method"] = "POST"
        batch_item["url"] = "/v1/chat/completions"
        batch_item["body"] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "response_format": {"type": "json_object"},
        }
        batch_item["body"]["messages"] = [{"role": "user", "content": item["prompt"]}]
        json_lines.append(batch_item)
    return json_lines


def run_eval(results, args):
    # try to load the existing results from args.eval_output_file
    if os.path.exists(args.eval_output_file) and not args.overwrite:
        cnt = 0
        with open(args.eval_output_file, "r") as f:
            existing_results = json.load(f)
        for i in range(len(existing_results)):
            e = existing_results[i]
            t = results[i]
            if e["prompt"] != t["prompt"]:
                continue
            # if e["prompt"] == t["prompt"] and e["result"] != "N/A":
            #     results[i]["result"] = e["result"]
            #     cnt += 1
            if "result" in e:
                t["result"] = e["result"]
                if "parsed_result" in e:
                    t["parsed_result"] = e["parsed_result"]
                    cnt += 1
            if "error" in e:
                t["error"] = e["error"]

            if args.mode == "pairwise" and "winner" in e:
                t["winner"] = e["winner"]

        print(f"loading {cnt} results from {args.eval_output_file}")

    # TODO: add support for using other APIs as judge
    openai_args = {
        "prompt": "TODO",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "json_mode": True,
        "stop": [],
    }
    if args.model:
        openai_args["model"] = args.model
    if args.engine:
        openai_args["engine"] = args.engine

    @retry_handler(retry_limit=10)
    def api(ind, item, **kwargs):
        model = kwargs["model"]
        if model.startswith("claude"):
            result = anthropic_chat_request(**kwargs)
        else:
            result = openai_chat_request(**kwargs)
        result = result[0]
        return result

    # results = results[args.start_idx:args.end_idx] # for debug
    # import pdb; pdb.set_trace()
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.eval_output_file} "):
        computed = False
        if item["result"] != "N/A" and item.get("error", "N/A") == "N/A" and "parsed_result" in item:
            results[ind]["parsed_result"] = parse_result(
                results[ind]["result"], eval_mode=args.mode
            )  # redo the parsing
            results[ind]["parsed"] = True if results[ind]["parsed_result"] is not None else False
            computed = True
            continue

        openai_args["prompt"] = item["prompt"]
        # if True:
        try:
            if not computed:
                result = api(ind, item, **openai_args)
                results[ind]["result"] = result
            else:
                result = results[ind]["result"]

            results[ind]["parsed_result"] = parse_result(results[ind]["result"])
            results[ind]["parsed"] = True if results[ind]["parsed_result"] is not None else False
            # r = results[ind]["parsed_result"]
            # if args.mode == "pairwise":
            #     # if r["choice"] in ["A", "B"]:
            #     #     results[ind]["winner"] = item["assignment"][r["choice"]]
            #     # elif r["choice"] == "tie":
            #     #     results[ind]["winner"] = "tie"
            #     # else:
            #     #     results[ind]["winner"] = r["choice"]
            #     pass  # Note that we will do the parsing later.
            # elif args.mode == "checklist":
            #     results[ind]["score"] = float(r["score"])
            # elif args.mode == "score":
            #     pass
            # if not args.model.startswith('claude'):
            # results[ind]["price"] = compute_cost(args.model, item["prompt"], results[ind]["result"])
            # else:
            #     results[ind]["price"] = {"cost": 0, "in_tokens": 0, "out_tokens": 0}
            results[ind]["error"] = "N/A"
        except Exception as e:
            # print(e)
            results[ind]["error"] = str(e)
            results[ind]["result"] = result
            results[ind]["parsed_result"] = {"choice": "N/A"}
            # results[ind]["price"] = {"cost": 0, "in_tokens": 0, "out_tokens": 0}
            pass

        # print("Done!")
        if ind % args.save_interval == 0 or ind == len(results) - 1:
            with open(args.eval_output_file, "w") as f:
                json.dump(results, f, indent=2)
    with open(args.eval_output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results


def shorten(text, K=-1):
    if K > 0 and len(text.split(" ")) > K:
        text = " ".join(text.split(" ")[:K]) + "... (truncated)"
    return text


def placeholder_generation(args, candidates, references, histories, last_queries, checklists):
    with open(args.eval_template) as f:
        eval_template = f.read()
        print(f"Loaded the eval_template from {args.eval_template}")

    results = []

    assert len(candidates) == len(references)

    L = len(candidates)
    if args.end_idx < 0 or args.end_idx > L:
        args.end_idx = L

    print(f"# examples in candidates: {len(candidates)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[args.start_idx : args.end_idx]
    references = references[args.start_idx : args.end_idx]
    histories = histories[args.start_idx : args.end_idx]
    last_queries = last_queries[args.start_idx : args.end_idx]
    checklists = checklists[args.start_idx : args.end_idx]

    results = []
    for item, ref_item, history, last_query, checklist in zip(
        candidates, references, histories, last_queries, checklists
    ):
        # print(item, ref_item, history, last_query, checklist)
        o = item["output"][0] if type(item["output"]) == list else item["output"]

        # random decide which is A and which is B
        d = {}
        d["session_id"] = item["session_id"]
        d["history"] = history
        d["last_query"] = last_query
        d["model_output"] = item["output"]
        # d["generator"] = args.target_model_name
        d["generator"] = item["generator"]
        if args.mode == "pairwise":
            r = ref_item["output"][0] if type(ref_item["output"]) == list else ref_item["output"]
            d["ref_output"] = r
            # d["ref_generator"] = args.ref_model_name
            d["ref_generator"] = ref_item["generator"]
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}

        ## Prompt composition for pairwise evaluation
        if args.mode == "pairwise":
            if random.random() < 0.5:
                A = o
                B = r
                d["assignment"] = {"A": d["generator"], "B": d["ref_generator"]}
            else:
                A = r
                B = o
                d["assignment"] = {"A": d["ref_generator"], "B": d["generator"]}
            prompt = eval_template
            prompt = prompt.replace("{$history}", shorten(history, args.max_words_to_eval))
            prompt = prompt.replace("{$user_query}", shorten(last_query, args.max_words_to_eval))
            A_output_str = shorten(A, args.max_words_to_eval)
            B_output_str = shorten(B, args.max_words_to_eval)
            if A_output_str.strip() == "":
                A_output_str = "[This model response is empty.]"
            if B_output_str.strip() == "":
                B_output_str = "[This model response is empty.]"
            prompt = prompt.replace("{$candidate_A}", A_output_str)
            prompt = prompt.replace("{$candidate_B}", B_output_str)
            checklist_mardkdown = ""
            for checklist_item in checklist:
                checklist_mardkdown += f"- {checklist_item}\n"
            prompt = prompt.replace("{$checklist}", checklist_mardkdown)
            d["prompt"] = prompt
            if A.strip() == "" and B.strip() == "":
                d["result"] = json.dumps({"reason": "Both responses are empty.", "choice": "A=B"})
            elif A.strip() == "":
                d["result"] = json.dumps({"reason": "The response A is empty.", "choice": "B++"})
            elif B.strip() == "":
                d["result"] = json.dumps({"reason": "The response B is empty.", "choice": "A++"})
            else:
                d["result"] = "N/A"
            results.append(d)

        elif args.mode == "score":
            prompt = eval_template
            prompt = prompt.replace("{$history}", shorten(history, args.max_words_to_eval))
            prompt = prompt.replace("{$user_query}", shorten(last_query, args.max_words_to_eval))
            prompt = prompt.replace("{$model_output}", shorten(o, args.max_words_to_eval))
            checklist_mardkdown = ""
            for checklist_item in checklist:
                checklist_mardkdown += f"- {checklist_item}\n"
            prompt = prompt.replace("{$checklist}", checklist_mardkdown)
            d_copy = d.copy()
            d_copy["prompt"] = prompt
            if o.strip() == "":
                d_copy["result"] = json.dumps(
                    {"strengths": "N/A", "weaknesses": "The model output is empty.", "score": "1"}
                )
            else:
                d_copy["result"] = "N/A"
            results.append(d_copy)

    return results


def compose_eval_item(b, t, r, histories, last_queries, checklists):
    if r is not None:
        assert b["session_id"] == t["session_id"] == r["session_id"]
    else:
        assert b["session_id"] == t["session_id"]
    history = ""
    checklist = b["checklist"]
    if len(b["conversation_input"]) > 0:
        for x in b["conversation_input"][:-1]:
            if x["role"] == "user":
                history += "USER: " + x["content"] + "\n\n"
            elif x["role"] == "assistant":
                history += "ASSISTANT: " + x["content"] + "\n\n"
    last_query = b["conversation_input"][-1]["content"]
    histories.append(history)
    last_queries.append(last_query)
    checklists.append(checklist)


def main():
    args = get_args()
    random.seed(args.seed)
    print(args)
    # assert the api key is ready
    if args.model.startswith("claude"):
        assert (
            os.environ.get("ANTHROPIC_API_KEY") is not None
        ), "Please set the ANTHROPIC_API_KEY in the environment variables."
    elif args.model.startswith("gpt"):
        assert (
            os.environ.get("OPENAI_API_KEY") is not None
        ), "Please set the OPENAI_API_KEY in the environment variables."

    if args.action.startswith("trial"):
        results = placeholder_generation(args)
        print(f"We have {len(results)} examples to evaluate!")
        with open(args.eval_output_file, "w") as f:
            json.dump(results, f, indent=2)
    elif args.action.startswith("eval"):
        if args.mode not in ["pairwise", "score"]:
            raise Exception("Not implemented yet!")

        bench_data = load_dataset(HF_BENCH_PATH, HF_BENCH_CONFIG, split="test")

        if args.local_result_file is not None:
            with open(args.local_result_file, "r") as f:
                target_model_data = json.load(f)
                print(f"Loaded the local results from {args.local_result_file}")
        else:
            try:
                target_model_data = load_dataset(HF_RESULTS_PATH, args.target_model_name, split="train")
            except Exception as e:
                print(f"Failed to load the target model data from {HF_RESULTS_PATH}/{args.target_model_name}")
                if args.local_result_file is None:
                    args.local_result_file = f"result_dirs/wild_bench_v2/{args.target_model_name}.json"
                print(f"Try loading from the local file {args.local_result_file}")
                with open(args.local_result_file, "r") as f:
                    target_model_data = json.load(f)
                    print(f"Loaded the local results from {args.local_result_file}")
        if args.mode == "pairwise":
            ref_model_data = load_dataset(HF_RESULTS_PATH, args.ref_model_name, split="train")
        else:
            print("No reference model is needed for checklist evaluation.")
            ref_model_data = [None] * len(target_model_data)
        histories = []
        last_queries = []
        checklists = []
        for b, t, r in zip(bench_data, target_model_data, ref_model_data):
            compose_eval_item(b, t, r, histories, last_queries, checklists)
        print(f"len(target_model_data)={len(target_model_data)}")
        print(f"len(ref_model_data)={len(ref_model_data)}")
        candidates = list(target_model_data)
        references = list(ref_model_data)
        results = placeholder_generation(args, candidates, references, histories, last_queries, checklists)
        if args.batch_mode:
            print("Batch mode is enabled!")
            json_lines = batch_eval_generate(results, args)
            with open(args.eval_output_file, "w") as f:
                for line in json_lines:
                    f.write(json.dumps(line) + "\n")
        else:
            results = run_eval(results, args)

    else:
        print("Not implemented yet!")


if __name__ == "__main__":
    main()
