from datasets import load_dataset, Dataset
from tqdm import tqdm
import asyncio
import os
import json
import tiktoken
import tempfile
import time
import os
from dcft.utils.api_request_parallel_processor import process_api_requests_from_file

niv2 = load_dataset("mlfoundations-dev/open-orca-niv2", split="train")
niv2 = niv2.select(range(10))

cot = load_dataset("mlfoundations-dev/open-orca-cot", split="train")
cot = cot.select(range(10))

LLM_JUDGE_SYSTEM_PROMPT = """
As an AI assistant, your role is to evaluate whether a given attempt correctly responds to the provided inputs by comparing it against the specified targets. You will be presented with three elements:

1. "inputs": The initial information or questions given.
2. "targets": The expected correct responses or outcomes.
3. "attempt": The response that needs to be evaluated.

Your task is to:

1. Carefully analyze the relationship between the inputs and targets.
2. Examine the attempt to see if it adequately addresses the inputs.
3. Compare the attempt to the targets, checking for accuracy and completeness.
4. Provide a brief explanation of your reasoning, highlighting any discrepancies or matches.
5. Conclude your response with a final judgment.

Your explanation should be clear and concise, focusing on the key points of comparison. After your explanation, end your response with a single word, either "yes" if the attempt correctly responds to the inputs by matching the targets, or "no" if it does not.

Remember, your final word must be either "yes" or "no", with no punctuation or additional text after it.
"""


def llm_judge_filter(
    dataset: Dataset,
    instruction_column: str,
    golden_answer_column: str,
    attempt_answer_column: str,
    filter: bool = False,
    job_path: str | None = None,
) -> Dataset:
    MODEL = "gpt-4o-mini"

    def create_jobs(samples, jobs_out_file):
        jobs = []
        idx_to_sample = {}
        for idx, sample in enumerate(tqdm(samples, desc="Creating jobs")):
            job = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"inputs: {sample[instruction_column]}\n\ntargets: {sample[golden_answer_column]}\n\nattempt: {sample[attempt_answer_column]}",
                    },
                ],
                "metadata": {"request_idx": idx},
            }
            jobs.append(job)
            idx_to_sample[idx] = sample

        with open(jobs_out_file, "w") as f:
            for job in jobs:
                f.write(json.dumps(job) + "\n")
        print(f"Jobs file {jobs_out_file} written to disk.")
        return jobs_out_file, idx_to_sample

    def run_online(samples, job_path, save_filepath):
        jobs_file, idx_to_sample = create_jobs(samples, job_path)

        print(f"Online generation with parallel processing starting, logging to {job_path}/output.log")
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=jobs_file,
                save_filepath=save_filepath,
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=30_000,
                max_tokens_per_minute=150_000_000,
                token_encoding_name=tiktoken.encoding_for_model(MODEL).name,
                max_attempts=5,
                resume=False,
            )
        )
        print(f"Parallel processing complete. Check {job_path}/output.log for details.")
        return idx_to_sample

    def process_output(output_file, idx_to_sample):
        filtered_count = 0
        total_count = 0
        processed_samples = []

        with open(output_file, "r") as f_in:
            for line in tqdm(f_in, desc="Processing samples"):
                response = json.loads(line)
                sample = idx_to_sample[response[2]["request_idx"]]
                classification = response[1]["choices"][0]["message"]["content"].strip().lower().split()[-1]
                classification = classification.rstrip(".").rstrip("!").strip('"')

                if classification not in ["yes", "no"]:
                    print(f"WARNING: Classification not yes or no for {sample[instruction_column]}: {classification}")

                sample["is_correct"] = classification == "yes"

                if sample["is_correct"]:
                    filtered_count += 1

                total_count += 1

                if not filter or sample["is_correct"]:
                    processed_samples.append(sample)

        # Calculate and print statistics
        filtered_percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0

        print("\nStatistics:")
        print(f"  Total documents: {total_count}")
        print(f"  Filtered: {filtered_count}")
        print(f"  Percentage of filtered: {filtered_percentage:.2f}%")

        return Dataset.from_list(processed_samples)

    # Use a temporary directory if job_path is None
    with tempfile.TemporaryDirectory() as temp_dir:
        actual_job_path = job_path or temp_dir
        os.makedirs(actual_job_path, exist_ok=True)
        timestamp = int(time.time())
        jobs_file = f"{actual_job_path}/jobs_{timestamp}.json"
        # Run the processing
        output_file = f"{actual_job_path}/output_{timestamp}.jsonl"
        idx_to_sample = run_online(dataset, jobs_file, output_file)
        processed_dataset = process_output(output_file, idx_to_sample)

    return processed_dataset


cot_filtered = llm_judge_filter(cot, "inputs", "targets", "model_response")
niv2_filtered = llm_judge_filter(niv2, "inputs", "targets", "model_response")

os.makedirs("dcft/slimorca/data", exist_ok=True)
cot_filtered.to_json("dcft/slimorca/data/cot_filtered.json")
niv2_filtered.to_json("dcft/slimorca/data/niv2_filtered.json")
