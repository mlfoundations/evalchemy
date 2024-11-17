from typing import Tuple, Callable, Optional
from datasets import Dataset
import aiofiles
import logging
import json
import os
import tiktoken
from tqdm import tqdm
import asyncio
import requests
from dcft.utils.api_request_parallel_processor import (
    process_api_requests_from_file,
    get_rate_limits,
    get_token_encoding_name,
    get_api_key,
)
from datasets.arrow_writer import ArrowWriter
import tempfile


def create_requests_file(dataset: Dataset, requests_file: str, create_requests_func: Callable, **kwargs) -> None:
    """
    Create a file containing API requests based on the given dataset.

    This function either loads existing jobs from the requests file or creates new requests
    using the provided function and writes them to the file.

    Args:
        dataset (Dataset): The input dataset to create requests from.
        requests_file (str): The path to the file where requests will be written or read from.
        create_requests_func (Callable): A function that creates requests from the dataset.
        **kwargs: Additional keyword arguments to pass to create_requests_func.


    """
    resume = True
    if os.path.exists(requests_file):
        if resume:
            logging.info(f"Loading existing jobs from {requests_file}")
            logging.info(f"Alternatively, delete the jobs file and re-run the annotator: `rm -rf {requests_file}`")
            # count existing jobs in file and print first job
            with open(requests_file, "r") as f:
                num_jobs = sum(1 for _ in f)
                f.seek(0)
                first_job = json.loads(f.readline())
            logging.info(f"Using existing jobs from {requests_file}")
            logging.info(f"Number of jobs: {num_jobs}")
            logging.info("Example job:")
            logging.info(json.dumps(first_job, indent=2))
        else:
            # Create new jobs and write to file
            error_message = (
                f"Existing job file {requests_file}. "
                f"Delete the jobs file and re-run the annotator: `rm -rf {requests_file}`. "
                f"Or run the annotator with the --resume flag to continue from the previous run."
            )
            raise ValueError(error_message)
    else:
        requests = create_requests_func(dataset, **kwargs)
        with open(requests_file, "w") as f:
            for request in tqdm(requests, desc="Writing requests to file"):
                f.write(json.dumps(request) + "\n")
        logging.info(f"Requests file {requests_file} written to disk.")


def run_online_generation(
    jobs_file: str, responses_file: str, model: str, resume_no_retry: bool, url: str, api_key: str
) -> None:
    """
    Run online generation of responses using parallel processing.

    This function sets up the rate limits, initiates the parallel processing of API requests,
    and handles the generation of responses.

    Args:
        jobs_file (str): The path to the file containing the API requests.
        responses_file (str): The path to the file where responses will be saved.
        model (str): The OpenAI model to use for generation.
        resume_no_retry (bool): Whether to resume without retrying failed requests.
        url (str): The URL of the API to use for generation.
        api_key (str): The API key to use for generation.
    """
    api_key = api_key or get_api_key(url)

    rpm, tpm = get_rate_limits(model, url, api_key)
    max_requests_per_minute = rpm
    print(f"Automatically set max_requests_per_minute to {rpm}")
    max_tokens_per_minute = tpm
    print(f"Automatically set max_tokens_per_minute to {tpm}")

    token_encoding_name = get_token_encoding_name(model)

    print(f"Online generation with parallel processing starting, logging to {jobs_file}/output.log")
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=jobs_file,
            save_filepath=responses_file,
            request_url=url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=5,
            resume=True,  # detects existing jobs and resume from there
            resume_no_retry=resume_no_retry,
        )
    )
    print(f"Parallel processing complete. Check {jobs_file}/output.log for details.")


def generate_response(
    dataset: Dataset,
    user_message_column: str,
    output_column: str = "model_response",
    system_message: Optional[str] = None,
    system_message_column: Optional[str] = None,
    working_dir: Optional[str] = None,
    model: str = "gpt-4o-mini",
    drop_columns: list[str] | None = None,
    resume_no_retry: bool = False,
    url: str = "https://api.openai.com/v1/chat/completions",
    api_key: str = None,
) -> Dataset:
    """
    Generate responses for a given dataset using the specified OpenAI model.

    This function handles the entire process of creating requests, running online generation,
    and processing the responses into a new dataset.

    Args:
        dataset (Dataset): The input dataset containing user messages.
        user_message_column (str): The name of the column containing user messages.
        system_message (Optional[str]): The system message to use for generation.
        system_message_column (Optional[str]): The name of the column containing system messages.
        output_column (str): The name of the column where generated responses will be stored.
        working_dir (Optional[str]): The directory to use for requests, responses, and dataset. If None, a temporary directory is created. This is critical for saving intermediate results.
        model (str): The OpenAI model to use for generation. Defaults to "gpt-4o-mini".
        url (str): The URL of the API to use for generation. Defaults to "https://api.openai.com/v1/chat/completions".
        api_key (str): The API key to use for generation. Defaults to the value of the OPENAI_API_KEY environment variable.

    Returns:
        Dataset: A new dataset containing the original data and the generated responses.

    Raises:
        ValueError: If all requests fail during processing.
        ValueError: If both system_message and system_message_column are specified.
    """

    def create_requests(dataset: Dataset, **kwargs) -> list:
        """
        Create a list of API requests from the given dataset.

        Args:
            dataset (Dataset): The input dataset.
            **kwargs: Additional keyword arguments, including user_message_column.

        Returns:
            list: A list of API request dictionaries.
        """
        user_message_column = kwargs.get("user_message_column")
        system_message = kwargs.get("system_message")
        system_message_column = kwargs.get("system_message_column")
        if system_message_column is not None and system_message is not None:
            raise ValueError("Cannot specify both system_message and system_message_column")
        requests = []
        # api parallel processing expects request_idx in metadata, I'm adding the sample so it's easy to process later
        for request_idx, sample in tqdm(enumerate(dataset), desc="Creating requests"):
            messages = []
            if system_message is not None:
                messages.append({"role": "system", "content": system_message})
            elif system_message_column is not None:
                messages.append({"role": "system", "content": sample[system_message_column]})
            messages.append({"role": "user", "content": sample[user_message_column]})
            request = {
                "model": model,
                "messages": messages,
                "metadata": {"sample": sample, "request_idx": request_idx},
            }
            requests.append(request)
        return requests

    def read_responses_file_and_write_to_dataset(
        responses_file: str, dataset_file: str, output_column: str, drop_columns: list[str] | None = None
    ) -> None:
        """
        Read responses from a file and write them to a new dataset.

        Args:
            responses_file (str): The path to the file containing API responses.
            dataset_file (str): The path to the output dataset file.
            output_column (str): The name of the column where generated responses will be stored.
            drop_columns (list[str] | None): List of column names to drop from the output dataset.

        Raises:
            ValueError: If all requests failed during processing.
        """
        total_count = 0
        failed_count = 0
        with ArrowWriter(path=dataset_file) as writer:
            with open(responses_file, "r") as f_in:
                for line in tqdm(f_in, desc="Reading responses and writing to dataset"):
                    total_count += 1
                    try:
                        response = json.loads(line)
                        if isinstance(response[1], list):
                            # this means that the request failed and we have a list of errors
                            logging.info(
                                f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}"
                            )
                            failed_count += 1
                            continue
                        metadata = response[2]
                        assistant_message = response[1]["choices"][0]["message"]["content"]
                        sample = metadata["sample"]
                        if drop_columns:  # drops specified columns
                            sample = {k: v for k, v in sample.items() if k not in drop_columns}
                        sample[output_column] = assistant_message
                        # note writer.write throws an error if the sample has null values, drop those columns
                        writer.write(sample)
                    except Exception as e:
                        logging.warning(f"Error: {e}")
                        logging.warning(f"Full response: {response}")
                        continue
            print(f"Read {total_count} responses, {failed_count} failed")
            print("Finalizing writer")
            if failed_count == total_count:
                raise ValueError("All requests failed")
            writer.finalize()

    with tempfile.TemporaryDirectory() as temp_dir:
        wdir = working_dir or temp_dir
        os.makedirs(working_dir, exist_ok=True)
        requests_file = f"{wdir}/requests.json"
        create_requests_file(
            dataset,
            requests_file,
            create_requests,
            user_message_column=user_message_column,
            system_message=system_message,
            system_message_column=system_message_column,
        )
        responses_file = f"{wdir}/responses.jsonl"
        run_online_generation(
            requests_file, responses_file, model, resume_no_retry=resume_no_retry, url=url, api_key=api_key
        )
        dataset_file = f"{wdir}/dataset.arrow"
        read_responses_file_and_write_to_dataset(responses_file, dataset_file, output_column, drop_columns)
        print("Dataset from file")
        if working_dir:
            processed_dataset = Dataset.from_file(dataset_file)
        else:
            # when you return the processed_dataset variable, it needs to have the original dataset_file retained
            # "Unlike load_dataset(), Dataset.from_file() memory maps the Arrow file without preparing the dataset in the cache, saving you disk space. The cache directory to store intermediate processing results will be the Arrow file directory in that case."
            # This will only work if the dataset is small enough to fit in memory
            # We should consider not allowing a temp dir as an option
            processed_dataset = Dataset.from_file(dataset_file, in_memory=True)

        print("Dataset loaded")
        print(processed_dataset)

    return processed_dataset


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
    judgement_decision_column: str = "model_judgement",
    judgement_reasoning_column: str = "model_judgement_full",
    filter: bool = False,
    working_dir: str | None = None,
    model: str = "gpt-4o-mini",
    url: str = "https://api.openai.com/v1/chat/completions",
    api_key: str = None,
) -> Dataset:
    """Filter dataset based on LLM judgements of response quality.

    Args:
        dataset (Dataset): Input dataset
        instruction_column (str): Column containing instructions
        golden_answer_column (str): Column containing correct answers
        attempt_answer_column (str): Column containing attempted answers
        judgement_decision_column (str, optional): Column to store judgement decisions. Defaults to "model_judgement"
        judgement_reasoning_column (str, optional): Column to store judgement reasoning. Defaults to "model_judgement_full"
        filter (bool, optional): Whether to filter out failed attempts. Defaults to False
        working_dir (str | None, optional): Directory for temporary files. Defaults to None
        model (str, optional): Model to use. Defaults to "gpt-4o-mini"

    Returns:
        Dataset: Dataset with judgements (and optionally filtered)
    """

    def create_requests(dataset: Dataset, **kwargs) -> dict:
        instruction_column = kwargs.get("instruction_column")
        golden_answer_column = kwargs.get("golden_answer_column")
        attempt_answer_column = kwargs.get("attempt_answer_column")
        requests = []
        # api parallel processing expects request_idx in metadata, I'm adding the sample so it's easy to process later
        for request_idx, sample in tqdm(enumerate(dataset), desc="Creating requests"):
            request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"inputs: {sample[instruction_column]}\n\ntargets: {sample[golden_answer_column]}\n\nattempt: {sample[attempt_answer_column]}",
                    },
                ],
                "metadata": {"sample": sample, "request_idx": request_idx},
            }
            requests.append(request)
        return requests

    def read_responses_file_and_write_to_dataset(responses_file, dataset_file):
        filtered_count = 0
        total_count = 0
        failed_count = 0
        with ArrowWriter(path=dataset_file) as writer:
            with open(responses_file, "r") as f_in:
                for line in tqdm(f_in, desc="Reading responses and writing to dataset"):
                    total_count += 1
                    try:
                        response = json.loads(line)
                        if isinstance(response[1], list):
                            # this means that the request failed and we have a list of errors
                            logging.info(
                                f"Request {response[2].get('request_idx')} failed due to errors: {response[1]}"
                            )
                            failed_count += 1
                            continue

                        metadata = response[2]
                        assistant_message = response[1]["choices"][0]["message"]["content"]
                        sample = metadata["sample"]
                        decision_word = assistant_message.strip().lower().split()[-1]
                        decision_word = "".join(char for char in decision_word if char.isalpha())
                        decision = decision_word == "yes"

                        sample[judgement_reasoning_column] = assistant_message
                        sample[judgement_decision_column] = decision
                        if decision_word not in ["yes", "no"]:
                            print(f"WARNING: Defaulting to False for classification '{decision_word}'")

                        if decision or not filter:
                            writer.write(sample)

                        filtered_count += int(not decision)
                        total_count += 1
                    except Exception as e:
                        logging.warning(f"Error: {e}")
                        logging.warning(f"Full response: {response}")
                        continue
            print(f"Read {total_count} responses, {failed_count} failed")
            print("Finalizing writer")
            writer.finalize()

        # Calculate and print statistics
        filtered_percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
        print("\nStatistics:")
        print(f"  Total samples: {total_count}")
        print(f"  Filtered: {filtered_count}")
        print(f"  Percentage filtered: {filtered_percentage:.2f}%")

    with tempfile.TemporaryDirectory() as temp_dir:
        wdir = working_dir or temp_dir
        os.makedirs(wdir, exist_ok=True)
        requests_file = f"{wdir}/requests.json"
        create_requests_file(
            dataset,
            requests_file,
            create_requests,
            instruction_column=instruction_column,
            golden_answer_column=golden_answer_column,
            attempt_answer_column=attempt_answer_column,
        )
        responses_file = f"{wdir}/responses.jsonl"
        run_online_generation(requests_file, responses_file, model, url=url, api_key=api_key)
        dataset_file = f"{wdir}/dataset.arrow"
        read_responses_file_and_write_to_dataset(responses_file, dataset_file)
        print("Dataset from file")
        if working_dir:
            processed_dataset = Dataset.from_file(dataset_file)
        else:
            # This will only work if the dataset is small enough to fit in memory
            # We should consider not allowing a temp dir as an option
            processed_dataset = Dataset.from_file(dataset_file, in_memory=True)
        print("Dataset loaded")
        print(processed_dataset)

    return processed_dataset
