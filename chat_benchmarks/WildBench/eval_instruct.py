import os
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import sys
import jsonlines

from datasets import load_dataset
from openai import OpenAI
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

sys.path.insert(0, "eval/chat_benchmarks/WildBench/src")

from src.unified_utils import save_outputs
from src.eval import (
    compose_eval_item,
    batch_eval_generate,
    placeholder_generation,
)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and data processing."""
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

def format_eval_file(submit_file: str, eval_file: str) -> str:
    """
    Format evaluation results from a submission file into a structured output file.

    Args:
        submit_file: Path to the submission file containing raw evaluation data.
        eval_file: Path to the evaluation file used to generate the output filename.

    Returns:
        Path to the formatted output file.
    """
    mode = "score"
    custom_id_to_submission: Dict[str, Dict[str, Any]] = {}
    with jsonlines.open(submit_file, "r") as f:
        for line in f:
            custom_id_to_submission[line["custom_id"]] = line

    results_json: List[Dict[str, Any]] = []
    with jsonlines.open(submit_file, "r") as f:
        for item in f:
            custom_id: str = item["custom_id"]
            submission: Dict[str, Any] = custom_id_to_submission[custom_id]
            custom_id_splits: List[str] = custom_id.split("||")
            session_id: str = custom_id_splits[0]
            try:
                eval_output_parsed: Dict[str, Any] = json.loads(item["response"]["choices"][0]["message"]["content"])
            except Exception:
                continue

            results_item: Dict[str, Any] = {
                "session_id": session_id,
                "parsed_result": eval_output_parsed,
            }

            prompt: str = submission["body"]["messages"][0]["content"]

            if mode == "score":
                process_score(results_item, custom_id_splits, eval_output_parsed, prompt)

            results_json.append(results_item)

    output_file: str = eval_file.replace("batch_results.jsonl", "scores.json")
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Output file written to {output_file}")
    return output_file

def process_score(
    results_item: Dict[str, Any],
    custom_id_splits: List[str],
    eval_output_parsed: Dict[str, Any],
    prompt: str
) -> None:
    """
    Process and update the results item with score information.

    Args:
        results_item: Dictionary to store the processed results.
        custom_id_splits: List of custom ID components.
        eval_output_parsed: Parsed evaluation output.
        prompt: The original prompt used for evaluation.
    """
    model_test: str = custom_id_splits[1]
    if "score" not in eval_output_parsed:
        return

    score: float = eval_output_parsed["score"]
    results_item.update({
        "model_test": model_test,
        "score": score,
    })

    model_output: str = prompt.split("<|begin_of_response|>\n")[1].split("<|end_of_response|>\n")[0]
    results_item["model_output"] = model_output.strip()

def process_file(eval_file: str, client: OpenAI) -> None:
    """
    Process an evaluation file using the OpenAI API client.

    Args:
        eval_file: Path to the evaluation file to be processed.
        client: An instance of the OpenAI API client.
    """
    with open(eval_file, "r") as file:
        lines = file.readlines()

    results = []
    for line in lines:
        payload = json.loads(line)
        payload["body"]["max_tokens"] = 4096
        response = client.chat.completions.create(**(payload["body"]))

        res = payload.copy()
        res["response"] = json.loads(response.json())
        results.append(res)

    output_file = eval_file.replace("batch-submit.jsonl", "results.jsonl")
    with open(output_file, "w") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

    print(f"Processing complete. Results saved to {output_file}")

def eval_instruct(model: LM) -> Dict[str, str]:
    """
    Perform instruction-based evaluation on a given language model.

    Args:
        model: The language model to be evaluated.
        config: Configuration for the evaluation.

    Returns:
        A dictionary containing the evaluation results and file paths.
    """
    config = EvaluationConfig()
    id_strs, chat_history, extracted_chats, metadata = load_eval_data(config)
    model_inputs = [model.apply_chat_template(chat) for chat in extracted_chats]
    temp_dir_obj: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
    temp_dir: str = temp_dir_obj.name
    filepath: str = os.path.join(temp_dir, "output.json")
    
    if config.end_index < 0 or config.end_index > len(model_inputs):
        config.end_index = len(model_inputs)
    
    model_inputs = model_inputs[config.start_index : config.end_index]
    id_strs = id_strs[config.start_index : config.end_index]
    chat_history = chat_history[config.start_index : config.end_index]
    metadata = {key: metadata[key][config.start_index : config.end_index] for key in metadata}
    
    print("Loading dataset ... done!")
    
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
    save_outputs(config, id_strs, outputs, chat_history, metadata, model_inputs, filepath)

    return {"filepath": filepath, "temp_dir_obj": temp_dir_obj}

def evaluate(results: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluate the results of a model's performance on various tasks.

    Args:
        results: A dictionary containing file paths and temporary directory information.
        config: Configuration for the evaluation.

    Returns:
        A dictionary containing various evaluation metrics and scores.
    """
    config = EvaluationConfig()

    temp_dir_obj = results["temp_dir_obj"]
    eval_folder = temp_dir_obj.name
    eval_file = f"{eval_folder}/batch-submit.jsonl"

    bench_data = load_dataset("allenai/WildBench", "v2", split="test")

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

    results = placeholder_generation(config, candidates, references, histories, last_queries, checklists)

    print("Batch mode is enabled!")
    json_lines = batch_eval_generate(results, config)
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
        task_mapping[item["id"]] = list(set(task_group_new[tag] for tag in tags))

    lengths = []
    scores = []
    task_cat_results = {}
    for item in eval_result:
        if "score" not in item:
            print(item)
            continue
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
            task_cat_results.setdefault(tag, []).append(float(item["score"]))

    task_cat_score = {tag: (sum(scores) / len(scores) - 5) * 2 for tag, scores in task_cat_results.items()}
    
    weights_by_task = {
        "Creative Tasks": 0.5,
        "Planning & Reasoning": 1.25,
        "Math & Data Analysis": 1,
        "Information/Advice seeking": 0.75,
        "Coding & Debugging": 1.25,
    }
    
    task_macro_score = sum(task_cat_score[tag] * weights_by_task[tag] for tag in task_cat_score) / sum(weights_by_task.values())
    
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

def load_eval_data(config: EvaluationConfig) -> Tuple[List[str], List[Any], List[str], Dict[str, List[Any]]]:
    """
    Load evaluation data based on the specified configuration and dataset name.

    Args:
        config: Configuration object containing data loading parameters.

    Returns:
        A tuple containing:
            - id_strs: List of session IDs
            - chat_history: List of chat histories
            - extracted_chats: List of extracted chat inputs
            - metadata: Dictionary of additional metadata for each item
    """
    chat_history = []
    id_strs = []
    extracted_chats = []
    metadata = {}

    if config.data_name == "wild_bench":
        dataset = load_dataset("allenai/WildBench", "v2", split="test")
        metadata = {"session_id": [], "primary_tag": []}
    else:
        raise ValueError(f"Data name {config.data_name} not supported")

    print(f"Loaded {len(dataset)} examples from {config.data_name}")

    for item in dataset:
        extracted_chats.append(item["conversation_input"])
        chat_history.append(extracted_chats)
        id_strs.append(item["session_id"])

        for key in metadata:
            assert key in item, f"Key {key} not found in metadata"
            metadata[key].append(item[key])
    
    print("Start applying template")
    return id_strs, chat_history, extracted_chats, metadata

