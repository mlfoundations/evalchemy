import sys

sys.path.insert(0, "eval/chat_benchmarks/WildBench/src")
from src.unified_infer import sanitize_args
from src.unified_utils import load_eval_data, save_outputs
from src.eval import (
    compose_eval_item,
    batch_eval_generate,
    placeholder_generation,
    HF_BENCH_PATH,
    HF_BENCH_CONFIG,
    HF_RESULTS_PATH,
)
import os
import tempfile
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass
import sys
import json
from datasets import load_dataset
import time
from openai import OpenAI
import json
import sys
import jsonlines
from typing import Dict, List, Tuple, Optional, Any
import json
import sys
import jsonlines
import copy


def format_eval_file(submit_file, eval_file) -> str:
    mode = "score"
    # Load submit data
    custom_id_to_submission: Dict[str, Dict[str, Any]] = {}
    with jsonlines.open(submit_file, "r") as f:
        for line in f:
            custom_id_to_submission[line["custom_id"]] = line

    # Process results
    results_json: List[Dict[str, Any]] = []
    with jsonlines.open(submit_file, "r") as f:
        for item in f:
            custom_id: str = item["custom_id"]
            submission: Dict[str, Any] = custom_id_to_submission[custom_id]
            custom_id_splits: List[str] = custom_id.split("||")
            session_id: str = custom_id_splits[0]
            try:
                eval_output_parsed: Dict[str, Any] = json.loads(item["response"]["choices"][0]["message"]["content"])
            except Exception as e:
                # print(f"Error parsing eval_output.")
                # eval_output_parsed = eval_output
                continue

            results_item: Dict[str, Any] = {
                "session_id": session_id,
                "parsed_result": eval_output_parsed,
            }

            prompt: str = submission["body"]["messages"][0]["content"]

            if mode == "pairwise":
                process_pairwise(results_item, custom_id_splits, eval_output_parsed, prompt)
            elif mode == "score":
                process_score(results_item, custom_id_splits, eval_output_parsed, prompt)

            results_json.append(results_item)

    # Write output file
    output_file: str = eval_file.replace("batch_results.jsonl", "scores.json")
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Output file written to {output_file}")
    return output_file


def process_pairwise(
    results_item: Dict[str, Any], custom_id_splits: List[str], eval_output_parsed: Dict[str, Any], prompt: str
) -> None:
    model_A: str = custom_id_splits[1].replace("A:", "")
    model_B: str = custom_id_splits[2].replace("B:", "")

    if "choice" not in eval_output_parsed:
        return

    choice: str = eval_output_parsed["choice"]
    winner, extent = get_winner_and_extent(choice, model_A, model_B)

    results_item.update(
        {
            "model_A": model_A,
            "model_B": model_B,
            "winner": winner,
            "extent": extent,
        }
    )

    model_A_output: str = prompt.split("<|begin_of_response_A|>\n")[1].split("<|end_of_response_A|>\n")[0]
    model_B_output: str = prompt.split("<|begin_of_response_B|>")[1].split("<|end_of_response_B|>\n")[0]
    results_item["model_outputs"] = {
        model_A: model_A_output.strip(),
        model_B: model_B_output.strip(),
    }


def process_score(
    results_item: Dict[str, Any], custom_id_splits: List[str], eval_output_parsed: Dict[str, Any], prompt: str
) -> None:
    model_test: str = custom_id_splits[1]
    if "score" not in eval_output_parsed:
        return

    score: float = eval_output_parsed["score"]
    results_item.update(
        {
            "model_test": model_test,
            "score": score,
        }
    )

    model_output: str = prompt.split("<|begin_of_response|>\n")[1].split("<|end_of_response|>\n")[0]
    results_item["model_output"] = model_output.strip()


def get_winner_and_extent(choice: str, model_A: str, model_B: str) -> Tuple[Optional[str], Optional[int]]:
    if choice == "A=B":
        return "tie", 0
    elif choice == "A+":
        return model_A, 1
    elif choice == "A++":
        return model_A, 2
    elif choice == "B+":
        return model_B, 1
    elif choice == "B++":
        return model_B, 2
    else:
        print(f"Error: choice {choice} not recognized.")
        return None, None


def process_file(eval_file, client):
    # Read the input file
    with open(eval_file, "r") as file:
        lines = file.readlines()

    results = []
    for line in lines:
        payload = json.loads(line)
        # Send the request using the payload directly
        payload["body"]["max_tokens"] = 4096
        response = client.chat.completions.create(**(payload["body"]))

        res = copy.deepcopy(payload)
        res["response"] = json.loads(response.json())
        results.append(res)

    # Save results
    output_file = eval_file.replace("batch-submit.jsonl", "results.jsonl")
    with open(output_file, "w") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

    print(f"Processing complete. Results saved to {output_file}")


@dataclass
class Config:
    engine: str = "vllm"
    output_folder: str = "./result_dirs/wild_bench/"
    download_dir: Optional[str] = None
    model_name: Optional[str] = "llama-3-instruct"
    model_pretty_name: Optional[str] = "lmevalharness"
    tokenizer_name: str = "auto"
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    tokenizer_mode: str = "auto"
    data_name: str = "wild_bench"
    batch_size: int = 1
    num_outputs: int = 1
    top_p: float = 1.0
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 7500
    max_model_len: Optional[int] = None
    start_index: int = 0
    end_index: int = -1
    filepath: str = "auto"
    overwrite: bool = False
    no_repeat_ngram_size: int = 0
    hf_bf16: bool = False
    hf_gptq: bool = False
    gpu_memory_utilization: float = 0.9
    use_hf_conv_template: bool = False
    use_imend_stop: bool = False
    mt_turn: int = -1
    mt_turn1_result: Optional[str] = None


def eval_instruct(model: LM) -> Dict[str, str]:
    # Data loading
    args = Config()
    args = sanitize_args(args)

    id_strs: List[str]
    chat_history: List[Any]
    model_inputs: List[str]
    metadata: Dict[str, List[Any]]
    id_strs, chat_history, extracted_chats, metadata = changed_load_eval_data(args)
    model_inputs = [model.apply_chat_template(chat) for chat in extracted_chats]
    temp_dir_obj: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
    temp_dir: str = temp_dir_obj.name
    filepath: str = os.path.join(temp_dir, "output.json")
    if args.end_index < 0 or args.end_index > len(model_inputs):
        args.end_index = len(model_inputs)
    model_inputs = model_inputs[args.start_index : args.end_index]
    id_strs = id_strs[args.start_index : args.end_index]
    chat_history = chat_history[args.start_index : args.end_index]
    metadata = {key: metadata[key][args.start_index : args.end_index] for key in metadata}
    print("loading dataset ... done!")
    all_instances: List[Instance] = [
        Instance(
            "generate_until",
            None,
            (
                inputs,
                {
                    "max_new_tokens": 1024,
                    "do_sample": False,
                },
            ),
            idx,
        )
        for idx, inputs in enumerate(model_inputs)
    ]

    outputs: List[str] = model.generate_until(all_instances)
    outputs = [[output] for output in outputs]
    save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    results: Dict[str, str] = {"filepath": filepath, "temp_dir_obj": temp_dir_obj}
    return results


def evaluate(results: Dict[str, str]) -> Dict[str, Any]:

    gpt_eval_name = "gpt-4o-2024-05-13"
    args = get_args()
    temp_dir_obj = results["temp_dir_obj"]
    eval_folder = results["temp_dir_obj"].name
    eval_file = f"{eval_folder}/batch-submit.jsonl"

    if os.path.exists(eval_file):
        print(f"File {eval_file} exists, skip generation")
        return

    bench_data = load_dataset(HF_BENCH_PATH, HF_BENCH_CONFIG, split="test")

    with open(results["filepath"], "r") as f:
        target_model_data = json.load(f)
        print(f"Loaded the local results from {results['filepath']}")

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

    print("Batch mode is enabled!")
    json_lines = batch_eval_generate(results, args)
    with open(eval_file, "w") as f:
        for line in json_lines:
            f.write(json.dumps(line) + "\n")
    client = OpenAI()
    process_file(eval_file, client)
    submit_file = f"{eval_folder}/results.jsonl"
    output_file = format_eval_file(submit_file, eval_file)
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

    with open(output_file, "r") as f:
        eval_result = json.load(f)
    task_mapping = {}
    wb_data = load_dataset("allenai/WildBench", "v2", split="test")
    for item in wb_data:

        tags = [item["primary_tag"]] + item["secondary_tags"]
        task_mapping[item["id"]] = []
        for tag in tags:
            task_mapping[item["id"]].append(task_group_new[tag])

        # deduplicate
        task_mapping[item["id"]] = list(set(task_mapping[item["id"]]))

    win_much_counts = []
    win_counts = []
    tie_counts = []
    lose_counts = []
    lose_much_counts = []
    lengths = []
    scores = []
    task_cat_results = {}
    task_cat_results = {}
    for item in eval_result:
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
    temp_dir_obj.cleanup()
    return {
        "score": sum(scores) / len(scores),
        "adjusted_score": (sum(scores) / len(scores) - 5) * 2,
        "task_macro_score": task_macro_score,
        "adjusted_task_macro_score": task_macro_score,
        "task_categorized_scores": task_cat_score,
        "total": len(eval_result),
        "avg_len": sum(lengths) / len(lengths),
    }


@dataclass
class EvaluationConfig:
    seed: int = 42
    model: str = "gpt-4o-2024-05-13"
    action: str = "eval"
    mode: str = "score"
    max_words_to_eval: int = -1
    eval_template: str = "eval/chat_benchmarks/WildBench/evaluation/eval_template.score.v2.md"
    target_model_name: str = ""
    eval_output_file: str = ""
    local_result_file: Optional[str] = None
    ref_model_name: Optional[str] = None
    batch_mode: bool = True
    start_idx: int = 0
    end_idx: int = -1
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 7500
    max_model_len: Optional[int] = None
    start_index: int = 0
    end_index: int = -1
    filepath: str = "auto"
    overwrite: bool = False
    no_repeat_ngram_size: int = 0
    hf_bf16: bool = False
    hf_gptq: bool = False
    gpu_memory_utilization: float = 0.9
    use_hf_conv_template: bool = False
    use_imend_stop: bool = False
    mt_turn: int = -1
    mt_turn1_result: Optional[str] = None


def get_args():
    # Create an instance of the configuration
    config = EvaluationConfig()

    return config


def changed_load_eval_data(args, data_name=None, model_name=None):
    if data_name is None:
        data_name = args.data_name
    if model_name is None:
        model_name = args.model_name
    chat_history = []
    id_strs = []
    extracted_chats = []
    metadata = {}
    if data_name == "wild_bench":
        # Note that this is changed to V2 on May 22.
        dataset = load_dataset("allenai/WildBench", "v2", split="test")
        metadata = {"session_id": [], "primary_tag": []}
    elif data_name == "wild_bench_v2_internal":
        dataset = load_dataset("WildEval/WildBench-v2-dev", split="test")
        metadata = {"session_id": [], "primary_tag": []}
    elif data_name == "wild_bench_v2_dev":
        dataset = load_dataset("WildEval/WildBench-V2", "v2.0522", split="test")
        metadata = {"session_id": [], "primary_tag": []}
    elif data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        metadata = {"dataset": []}
    elif data_name == "just_eval":
        dataset = load_dataset("re-align/just-eval-instruct", split="test")
        metadata = {"dataset": [], "source_id": []}
    elif data_name == "mt-bench":
        dataset = load_dataset(
            "json",
            data_files="https://huggingface.co/spaces/lmsys/mt-bench/raw/main/data/mt_bench/question.jsonl",
            split="train",
        )
        metadata = {"question_id": [], "category": []}
        if args.mt_turn == 2:
            with open(args.mt_turn1_result, "r") as f:
                mt_turn1_result = json.load(f)
            id_to_turn1_result = {}
            for item in mt_turn1_result:
                id_to_turn1_result[item["question_id"]] = item["turn1_output"]
    else:
        raise ValueError(f"Data name {data_name} not supported")

    print(f"Loaded {len(dataset)} examples from {data_name}")

    for ind, item in enumerate(dataset):
        extracted_chats.append(item["conversation_input"])
        chat_history.append(extracted_chats)
        id_strs.append(item["session_id"])

        for key in metadata:
            assert key in item, f"Key {key} not found in metadata"
            metadata[key].append(item[key])
    print("Start applying template")
    return id_strs, chat_history, extracted_chats, metadata
