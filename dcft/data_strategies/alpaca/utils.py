import json
import random
import multiprocessing
import os
import asyncio
import time
import tempfile
import tiktoken
import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from ctypes import c_bool
from datasets import Dataset
from openai import OpenAI
from dcft.data_strategies.alpaca.utils_alpaca_repo import *
from dcft.utils.api_request_parallel_processor import process_api_requests_from_file
import faiss
from sentence_transformers import SentenceTransformer


def load_seed_instructions(seed_tasks_path: str) -> Dataset:
    """
    Loads alpaca seed instruction tasks from a given JSON file path.

    Args:
        seed_tasks_path (str): Path to the JSON file containing the seed tasks.

    Returns:
        Dataset: A dataset containing the seed instructions, inputs, and outputs.
    """
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    seed_instructions_dataset = Dataset.from_pandas(pd.DataFrame(seed_instruction_data))
    return seed_instructions_dataset


def generate_from_openai(
    seed_tasks: Dataset,
    num_instructions_to_generate: int = 100,
    num_prompt_instructions: int = 3,
    model_name: str = "gpt-4o-mini",
) -> Dataset:
    """
    Generates new instructions using OpenAI API based on seed tasks.

    Args:
        seed_tasks (Dataset): The dataset containing seed instructions.
        num_instructions_to_generate (int, optional): Number of instructions to generate. Default is 100.
        num_prompt_instructions (int, optional): Number of instructions used to prompt the model. Default is 3.
        model_name (str, optional): Model name used for OpenAI API calls. Default is "gpt-4o-mini".

    Returns:
        Dataset: A dataset containing the generated instructions.
    """

    def create_jobs(seed_tasks: Dataset, jobs_out_file: str) -> str:
        num_cycles = num_instructions_to_generate // (20 - num_prompt_instructions) + 1
        jobs = []
        for idx in tqdm(range(num_cycles), desc="Creating jobs"):
            seed_idxs = random.sample(range(len(seed_tasks)), num_prompt_instructions)
            prompt_instructions = [seed_tasks[i] for i in seed_idxs]
            prompt = encode_prompt(prompt_instructions)
            job = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "metadata": {"request_idx": idx},
            }
            jobs.append(job)
        with open(jobs_out_file, "w") as f:
            for job in jobs:
                f.write(json.dumps(job) + "\n")
        print(f"Jobs file {jobs_out_file} written to disk.")
        return jobs_out_file

    def run_online(seed_tasks: Dataset, job_path: str, responses_path: str):
        jobs_file = create_jobs(seed_tasks, job_path)

        print(f"Online generation with parallel processing starting, logging to {job_path}/output.log")
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=jobs_file,
                save_filepath=responses_path,
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=30_000,
                max_tokens_per_minute=150_000_000,
                token_encoding_name=tiktoken.encoding_for_model(model_name).name,
                max_attempts=5,
                resume=False,
            )
        )
        print(f"Parallel processing complete. Check {job_path}/output.log for details.")

    def process_output(output_file: str, num_prompt_instructions: int) -> Dataset:
        processed_samples = []

        with open(output_file, "r") as f_in:
            for line in tqdm(f_in, desc="Processing samples"):
                response = json.loads(line)
                output_instructions = post_process_response(num_prompt_instructions, response)
                processed_samples.extend(output_instructions)

        processed_dataset = Dataset.from_list(processed_samples)
        return processed_dataset

    with tempfile.TemporaryDirectory() as temp_dir:
        job_path = temp_dir
        os.makedirs(job_path, exist_ok=True)
        timestamp = int(time.time())
        jobs_file = f"{job_path}/jobs_{timestamp}.json"
        responses_file = f"{job_path}/responses_{timestamp}.jsonl"
        run_online(seed_tasks, jobs_file, responses_file)
        processed_dataset = process_output(responses_file, num_prompt_instructions)

    return processed_dataset


def filter_instructions_heuristics(dataset: Dataset) -> Dataset:
    """
    Filters a dataset of instructions using various heuristics.

    Args:
        dataset (Dataset): The dataset to filter.

    Returns:
        Dataset: A dataset containing instructions that passed the heuristics filtering.
    """
    too_short = []
    too_long = []
    blacklisted = []
    write_a_program_prefix = []
    punctuation_prefix = []
    ascii_prefix = []
    not_filtered = []
    total_count = 0

    for instruction_dict in dataset:
        inst, input_text, output_text = (
            instruction_dict["instruction"].strip(),
            instruction_dict["input"].strip(),
            instruction_dict["output"].strip(),
        )
        total_count += 1

        # filter out too short or too long instructions
        if len(inst.split()) <= 3:
            too_short.append(
                {"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "too_short"}
            )
            continue

        if len(inst.split()) > 150:
            too_long.append(
                {"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "too_long"}
            )
            continue

        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        if any(find_word_in_string(word, inst) for word in blacklist):
            blacklisted.append(
                {"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "blacklisted"}
            )
            continue

        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit confusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            write_a_program_prefix.append(
                {
                    "instruction": inst,
                    "input": input_text,
                    "output": output_text,
                    "filtered_reason": "write_a_program_prefix",
                }
            )
            continue

        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            punctuation_prefix.append(
                {
                    "instruction": inst,
                    "input": input_text,
                    "output": output_text,
                    "filtered_reason": "punctuation_prefix",
                }
            )
            continue

        # filter those starting with non-english character
        if not inst[0].isascii():
            ascii_prefix.append(
                {"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "ascii_prefix"}
            )
            continue

        # If not filtered, add to not_filtered list
        not_filtered.append({"instruction": inst, "input": input_text, "output": output_text})

    not_filtered_dataset = Dataset.from_pandas(pd.DataFrame(not_filtered))
    print(not_filtered_dataset)
    return not_filtered_dataset


def filter_instructions_rouge_fast(
    dataset: Dataset, seed_instructions: Dataset, similarity_threshold: float = 0.7
) -> Dataset:
    """
    Filters instructions based on a similarity threshold using SentenceTransformer and FAISS for fast processing.

    Args:
        dataset (Dataset): The dataset to filter.
        seed_instructions (Dataset): A dataset of seed instructions for comparison.
        similarity_threshold (float, optional): Threshold for filtering based on cosine similarity. Default is 0.7.

    Returns:
        Dataset: A dataset containing the filtered instructions.
    """
    sentences = list(dataset["instruction"])

    # Step 1: Convert sentences to embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_numpy=True)
    # Step 2: Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    num_sentences = len(sentences)

    # Shared data structures for multiprocessing
    manager = multiprocessing.Manager()
    selected_indices = manager.list()
    visited_array = multiprocessing.Array(c_bool, num_sentences)
    lock = multiprocessing.Lock()

    def process_indices(index_range):
        # Each process creates its own FAISS index
        local_index = faiss.IndexFlatIP(embeddings.shape[1])
        local_index.add(embeddings)

        local_selected = []
        for i in index_range:
            with lock:
                if visited_array[i]:
                    continue
                visited_array[i] = True
                local_selected.append(i)

            # Search for similar sentences
            D, I = local_index.search(embeddings[i : i + 1], num_sentences)
            similar_indices = [idx for idx, score in zip(I[0], D[0]) if score > similarity_threshold and idx != i]

            with lock:
                for idx in similar_indices:
                    visited_array[idx] = True

        # Append local selected indices to the global list
        selected_indices.extend(local_selected)

    # Step 3: Parallel processing
    def parallel_greedy_selection():
        num_processes = multiprocessing.cpu_count()
        batch_size = (num_sentences + num_processes - 1) // num_processes  # Ceiling division

        processes = []
        for i in range(num_processes):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_sentences)
            index_range = range(start_idx, end_idx)
            p = multiprocessing.Process(target=process_indices, args=(index_range,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # Convert selected indices to a regular list
        final_selected_indices = list(selected_indices)
        return final_selected_indices

    # Run the parallel greedy selection
    selected_indices = parallel_greedy_selection()

    out = Dataset.from_pandas(pd.DataFrame(dataset).iloc[selected_indices].reset_index(drop=True))
    print(out)
    return out


def filter_instructions_rouge(
    dataset: Dataset, seed_instructions: Dataset, similarity_threshold: float = 0.7
) -> Dataset:
    """
    Filters instructions based on ROUGE-L score similarity to seed instructions.

    Args:
        dataset (Dataset): The dataset to filter.
        seed_instructions (Dataset): The dataset of seed instructions for comparison.

    Returns:
        Dataset: A dataset containing instructions that passed the ROUGE filtering.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    current_pool_instructions = list(seed_instructions["instruction"])
    current_pool_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in current_pool_instructions]
    not_filtered, filtered = [], []

    for instruction_dict in tqdm(dataset):
        inst, input_text, output_text = (
            instruction_dict["instruction"].strip(),
            instruction_dict["input"].strip(),
            instruction_dict["output"].strip(),
        )
        # computing similarity with the pre-tokenzied instructions
        new_instruction_tokens = scorer._tokenizer.tokenize(inst)

        rouge_scores = np.array(
            [
                rouge_scorer._score_lcs(new_instruction_tokens, tokens).fmeasure
                for tokens in current_pool_instruction_tokens
            ]
        )

        if max(rouge_scores) > similarity_threshold:
            filtered.append(
                {
                    "instruction": inst,
                    "input": input_text,
                    "output": output_text,
                    "filtered_reason": "rouge_similarity",
                    "most_similar_score": max(rouge_scores),
                    "most_similar_instruction": current_pool_instructions[np.argmax(rouge_scores)],
                }
            )
            continue

        not_filtered.append({"instruction": inst, "input": input_text, "output": output_text})
        current_pool_instructions.append(inst)
        current_pool_instruction_tokens.append(new_instruction_tokens)

    not_filtered_dataset = Dataset.from_pandas(pd.DataFrame(not_filtered))
    return not_filtered_dataset
