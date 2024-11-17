import asyncio
import json
import logging
import os
from math import ceil
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dcft.dataset.annotators import AnnotatorConfig
    from dcft.dataset.generation import GenerationConfig

import aiofiles
import tiktoken
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from dcft.dataset.annotators._baseannotator import BaseAnnotator
from dcft.utils.api_request_parallel_processor import process_api_requests_from_file


class GPTAnnotator(BaseAnnotator):
    def __init__(self, annotator_name: str, annotator_config: "AnnotatorConfig", **kwargs: Any) -> None:
        """
        Initializes the GPTAnnotator.

        Args:
            annotator_name (str): The name of the annotator.
            annotator_config (AnnotatorConfig): Configuration for the annotator.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(annotator_name, annotator_config, **kwargs)
        if self.config.batch and self.config.resume:
            raise ValueError("Batch mode and resume mode are not compatible.")
        self.client = OpenAI()
        self.encoder_name = tiktoken.encoding_for_model(annotator_name).name
        self.async_client = AsyncOpenAI()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def annotate(self, data: Any, generation_config: "GenerationConfig", temp_dir: str) -> None:
        """
        Annotates the provided data using the generation configuration.

        Args:
            data (Any): The data containing user prompts and system prompts.
            generation_config (GenerationConfig): Configuration for the generation process.
            temp_dir (str): Directory used to save logs and intermediate outputs
        """
        return asyncio.run(self.annotate_async(data, generation_config, temp_dir))

    async def annotate_async(self, data: Any, generation_config: "GenerationConfig", job_path: str) -> None:
        """
        Asynchronously annotates the provided data using the generation configuration.
        annotate is just a wrapper around this since the interface doesn't use aync right now

        Args:
            data (Any): The data containing user prompts and system prompts.
            generation_config (Any): Configuration for the generation process.
            temp_dir (str): Directory used to save logs and intermediate outputs
        """
        n = len(data.user_prompts)
        os.makedirs(job_path, exist_ok=True)

        if self.config.batch:
            num_batches = ceil(n / self.config.max_batch_size)
            batch_objects: List[Any] = []
            tasks: List[asyncio.Task] = []

            for i in range(num_batches):
                start_idx = i * self.config.max_batch_size
                end_idx = min((i + 1) * self.config.max_batch_size, n)

                jobs = self.create_jobs(end_idx - start_idx, data, generation_config, start_idx)
                task = self.write_jobs_and_submit_batch(jobs, job_path, i)
                tasks.append(task)

            batch_objects = await asyncio.gather(*tasks)
            data.batch_objects = batch_objects
        else:
            jobs_file = f"{job_path}/jobs.json"
            if os.path.exists(jobs_file):
                if self.config.resume:
                    self.logger.info(f"Resuming from previous run, loading existing jobs from {jobs_file}")
                    self.logger.info(
                        f"To regenerate the jobs file, delete the jobs file and re-run the annotator: `rm -rf {jobs_file}`"
                    )
                    # Load existing jobs from file
                    with open(jobs_file, "r") as f:
                        jobs = [json.loads(line) for line in f]
                    self.logger.info(f"Using existing jobs from {jobs_file}")
                    self.logger.info(f"Number of jobs: {len(jobs)}")
                    self.logger.info("Example job:")
                    self.logger.info(json.dumps(jobs[0], indent=2))
                else:
                    # Create new jobs and write to file
                    error_message = (
                        f"Existing job file {jobs_file}. "
                        f"Delete the jobs file and re-run the annotator: `rm -rf {jobs_file}`. "
                        f"Or run the annotator with the --resume flag to continue from the previous run."
                    )
                    raise ValueError(error_message)
            else:
                jobs = self.create_jobs(n, data, generation_config)
                async with aiofiles.open(jobs_file, "w") as f:
                    for job in jobs:
                        await f.write(json.dumps(job) + "\n")
                self.logger.info(f"Jobs file {jobs_file} written to disk.")

            await self.run_online(data, job_path, n)

    async def run_online(self, data: Any, job_path: str, n: int) -> None:
        """
        Runs the online generation with parallel processing.

        Args:
            data (Any): The data containing user prompts and system prompts.
            job_path (str): The path to save job-related files.
            n (int): The number of user prompts.
        """
        self.logger.info(f"Online generation with parallel processing starting, logging to {job_path}/output.log")
        await process_api_requests_from_file(
            requests_filepath=f"{job_path}/jobs.json",
            save_filepath=f"{job_path}/output.jsonl",
            request_url="https://api.openai.com/v1/chat/completions",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_requests_per_minute=float(self.config.max_requests_per_minute),
            max_tokens_per_minute=float(self.config.max_tokens_per_minute),
            token_encoding_name=self.encoder_name,
            max_attempts=5,
            resume=self.config.resume,
        )
        self.logger.info(f"Parallel processing complete. Check {job_path}/output.log for details.")
        # Load file that was created
        outputs: Dict[int, str] = {}
        with open(f"{job_path}/output.jsonl", "r") as f:
            for line in f:
                l = json.loads(line)
                outputs[l[2]["request_idx"]] = l[1]["choices"][0]["message"]["content"]
        self.logger.info(f"Number of outputs: {len(outputs)}")
        data.annotations = [outputs.get(i, {}) for i in range(n)]

    def create_job_dict(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_config: "GenerationConfig",
        batch: bool,
        idx: Optional[int],
    ) -> Dict[str, Any]:
        """
        Creates a job dictionary for the API request.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user prompt.
            generation_config (Any): Configuration for the generation process.
            batch (bool): Whether the request is part of a batch.
            idx (Optional[int]): The index of the job.

        Returns:
            Dict[str, Any]: The job dictionary for the API request.
        """
        message = [{"role": "user", "content": user_prompt}]
        if system_prompt != "":
            message = [{"role": "system", "content": system_prompt}] + message
        request_body = {
            "model": self.annotator_name,
            "messages": message,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "seed": generation_config.seed,
            "max_tokens": generation_config.max_tokens,
            "stop": generation_config.stop,
            "frequency_penalty": generation_config.frequency_penalty,
            "logit_bias": generation_config.logit_bias,
            "logprobs": generation_config.logprobs,
            "top_logprobs": generation_config.top_logprobs,
            "n": generation_config.n,
            "presence_penalty": generation_config.presence_penalty,
        }
        if batch:
            return {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body,
            }
        if idx is not None:
            request_body["metadata"] = {
                "request_idx": idx,
            }
        return request_body

    def create_jobs(self, n: int, data: Any, generation_config: Any, start_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Creates a list of jobs for the API requests.

        Args:
            n (int): The number of jobs to create.
            data (Any): The data containing user prompts and system prompts.
            generation_config (Any): Configuration for the generation process.
            start_idx (int): The starting index for the jobs.

        Returns:
            List[Dict[str, Any]]: A list of job dictionaries for the API requests.
        """
        self.logger.info(f"Creating {n} API request jobs")
        jobs: List[Dict[str, Any]] = []
        for idx in tqdm(range(n)):
            job = self.create_job_dict(
                data.system_prompts[start_idx + idx],
                data.user_prompts[start_idx + idx],
                generation_config,
                self.config.batch,
                start_idx + idx,
            )
            jobs.append(job)
        return jobs

    async def write_jobs_and_submit_batch(self, jobs: List[Dict[str, Any]], job_path: str, batch_index: int) -> Any:
        """
        Writes jobs to a file and submits the batch for processing.

        Args:
            jobs (List[Dict[str, Any]]): The list of job dictionaries.
            job_path (str): The path to save job-related files.
            batch_index (int): The index of the batch.

        Returns:
            Any: The batch object returned from the API.
        """
        jobs_file = f"{job_path}/jobs_batch_{batch_index}.json"

        # Write jobs to file
        async with aiofiles.open(jobs_file, "w") as f:
            for job in jobs:
                await f.write(json.dumps(job) + "\n")
        self.logger.info(f"Jobs file {jobs_file} written to disk.")

        # Submit batch
        self.logger.info(f"Batch generation starting for batch {batch_index}.")
        async with aiofiles.open(jobs_file, "rb") as file:
            file_content = await file.read()
            batch_input_file = await self.async_client.files.create(file=file_content, purpose="batch")

        batch_input_file_id = batch_input_file.id
        self.logger.info(f"File uploaded: {batch_input_file}")

        batch_object = await self.async_client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        self.logger.info(f"Batch request submitted, received batch object: {batch_object}")
        return batch_object
