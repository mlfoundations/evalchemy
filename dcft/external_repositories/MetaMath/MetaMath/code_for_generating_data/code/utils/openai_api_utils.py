import openai
import time
import os
from openai import OpenAI
import logging

client = OpenAI()
import tempfile
import json
import aiofiles
import asyncio
from tqdm import tqdm
import tiktoken

from dcft.utils.api_request_parallel_processor import process_api_requests_from_file
from dcft.dataset.reannotate import get_rate_limits

# openai.organization = ""


async def create_requests_file(requests, requests_file: str, **kwargs):
    resume = True
    if os.path.exists(requests_file):
        if resume:
            logging.info(f"Loading existing jobs from {requests_file}")
            logging.info(f"Alternatively, delete the jobs file and re-run the annotator: `rm -rf {requests_file}`")
            # Load existing jobs from file
            with open(requests_file, "r") as f:
                jobs = [json.loads(line) for line in f]
            logging.info(f"Using existing jobs from {requests_file}")
            logging.info(f"Number of jobs: {len(jobs)}")
            logging.info("Example job:")
            logging.info(json.dumps(jobs[0], indent=2))
        else:
            # Create new jobs and write to file
            error_message = (
                f"Existing job file {requests_file}. "
                f"Delete the jobs file and re-run the annotator: `rm -rf {requests_file}`. "
                f"Or run the annotator with the --resume flag to continue from the previous run."
            )
            raise ValueError(error_message)
    else:
        async with aiofiles.open(requests_file, "w") as f:
            for request in requests:
                await f.write(json.dumps(request) + "\n")
        logging.info(f"Requests file {requests_file} written to disk.")


def create_response_chat_batch(prompt_inputs, eng="gpt-3.5-turbo", temperature=0.0, timeout=20):
    requests = [
        {"model": eng, "messages": prompt_input, "temperature": temperature, "metadata": {"request_idx": idx}}
        for idx, prompt_input in enumerate(prompt_inputs)
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = temp_dir
        os.makedirs(working_dir, exist_ok=True)
        requests_file = f"{working_dir}/requests.json"
        asyncio.run(create_requests_file(requests, requests_file))
        output_file = f"{working_dir}/responses.jsonl"
        run_online_generation(requests_file, output_file, eng)
        responses = read_responses_file(output_file)
        messages = [assistant_message for assistant_message, metadata in responses]

    return messages


def read_responses_file(output_file):
    valid_responses = []
    total_count = 0
    failed_count = 0
    with open(output_file, "r") as f_in:
        for line in tqdm(f_in, desc="Reading responses"):
            total_count += 1
            try:
                response = json.loads(line)
                if isinstance(response[1], list):
                    # this means that the request failed and we have a list of errors
                    logging.info(f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}")
                    failed_count += 1
                    continue

                metadata = response[2]
                assistant_message = response[1]["choices"][0]["message"]["content"]
                valid_responses.append((assistant_message, metadata))

            except Exception as e:
                logging.warning(f"Error: {e}")
                logging.warning(f"Full response: {response}")
                continue
    print(f"Read {total_count} responses, {failed_count} failed")
    return valid_responses


def run_online_generation(jobs_file, responses_file, model):
    rpm, tpm = get_rate_limits(model)
    max_requests_per_minute = rpm
    print(f"Automatically set max_requests_per_minute to {rpm}")
    max_tokens_per_minute = tpm
    print(f"Automatically set max_tokens_per_minute to {tpm}")

    print(f"Online generation with parallel processing starting, logging to {jobs_file}/output.log")
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=jobs_file,
            save_filepath=responses_file,
            request_url="https://api.openai.com/v1/chat/completions",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=tiktoken.encoding_for_model(model).name,
            max_attempts=5,
            resume=True,  # detects existing jobs and resume from there
        )
    )
    print(f"Parallel processing complete. Check {jobs_file}/output.log for details.")
