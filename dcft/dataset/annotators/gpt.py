import asyncio
import json
import logging
import os
import uuid

import tiktoken
from openai import OpenAI
from tqdm import tqdm

from dcft.dataset.annotators._baseannotator import BaseAnnotator
from dcft.utils.api_request_parallel_processor import process_api_requests_from_file


class GPTAnnotator(BaseAnnotator):
    def __init__(self, annotator_name, annotator_config, **kwargs):
        super().__init__(annotator_name, annotator_config, **kwargs)
        if self.config.batch and self.config.resume:
            raise ValueError("Batch mode and resume mode are not compatible.")
        self.client = OpenAI()
        self.encoder_name = tiktoken.encoding_for_model(annotator_name).name

    def create_job_dict(self, prompt, generation_config, batch, idx):
        request_body = {
            "model": self.annotator_name,
            "messages": [{"role": "user", "content": prompt}],
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

    def _create_and_write_jobs(self, n, data, generation_config, jobs_file):
        """
        Create job dictionaries and write them to the jobs file.

        Args:
            n (int): Number of jobs to create
            data (object): Data object containing user prompts
            generation_config (object): Configuration for job generation
            jobs_file (str): Path to the jobs file

        Returns:
            list: List of created job dictionaries
        """
        print(f"Templating {n} API requests jobs")
        jobs = []
        for idx in tqdm(range(n)):
            job = self.create_job_dict(data.user_prompts[idx], generation_config, self.config.batch, idx)
            jobs.append(job)

        with open(jobs_file, "w") as f:
            for job in jobs:
                f.write(json.dumps(job) + "\n")
        print(f"Jobs file {jobs_file} written to disk.")

        return jobs

    def annotate(self, data, generation_config):
        n = len(data.user_prompts)

        # Create jobs.json for parallel request processing
        job_path = f"datasets/temp/{data.data_path.replace('/', '_')}"
        os.makedirs(job_path, exist_ok=True)
        jobs_file = f"{job_path}/jobs_batch.json" if self.config.batch else f"{job_path}/jobs.json"

        # Check if the jobs file already exists
        if os.path.exists(jobs_file):
            if self.config.resume:
                print(f"Resuming from previous run, loading existing jobs from {jobs_file}")
                print(
                    f"To regenerate the jobs file, delete the jobs file and re-run the annotator: `rm -rf {jobs_file}`"
                )
                # Load existing jobs from file
                with open(jobs_file, "r") as f:
                    jobs = [json.loads(line) for line in f]
                print(f"Using existing jobs from {jobs_file}")
                print(f"Number of jobs: {len(jobs)}")
                print("Example job:")
                print(json.dumps(jobs[0], indent=2))
            else:
                # Create new jobs and write to file
                error_message = (
                    f"Existing job file {jobs_file}. "
                    f"Delete the jobs file and re-run the annotator: `rm -rf {jobs_file}`. "
                    f"Or run the annotator with the --resume flag to continue from the previous run."
                )
                raise ValueError(error_message)

        jobs = self._create_and_write_jobs(n, data, generation_config, jobs_file)

        if self.config.batch:
            self.run_batch(data, job_path)
        else:
            self.run_online(data, job_path, n)

    def run_batch(self, data, job_path):
        print(f"Batch generation starting.")
        batch_input_file = self.client.files.create(
            file=open(f"{job_path}/jobs_batch.json", "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        print(f"File uploaded: {batch_input_file}")

        batch_object = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"Batch request submitted, received batch object: " f"{batch_object}")
        data.batch_object = batch_object

    def run_online(self, data, job_path, n):
        print(
            f"Online generation with parallel processing starting, "
            f"logging to {job_path}/output.log"
        )
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=f"{job_path}/jobs.json",
                save_filepath=f"{job_path}/output.jsonl",
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=float(self.config.max_requests_per_minute),
                max_tokens_per_minute=float(self.config.max_tokens_per_minute),
                token_encoding_name=self.encoder_name,
                max_attempts=5,
                logging_level=logging.INFO,
                resume=self.config.resume,
                log_filepath=f"{job_path}/output.log",
            )
        )
        print(f"Parallel processing complete. Check {job_path}/output.log for details.")
        # Load file that was created
        outputs = {}
        with open(f"{job_path}/output.jsonl", "r") as f:
            for line in f:
                l = json.loads(line)
                outputs[l[2]["request_idx"]] = l[1]["choices"][0]["message"]["content"]
        print(f"Number of outputs: {len(outputs)}")
        data.annotations = [outputs.get(i, {}) for i in range(n)]
