import os
import json
from tqdm import tqdm
import tiktoken
from openai import OpenAI
from dcft.dataset.annotators._baseannotator import BaseAnnotator
import asyncio
from dcft.utils.api_request_parallel_processor import process_api_requests_from_file


class GPTAnnotator(BaseAnnotator):
    def __init__(self, annotator_name, annotator_config, **kwargs):
        super().__init__(annotator_name, annotator_config, **kwargs)
        self.client = OpenAI()
        self.encoder_name = tiktoken.encoding_for_model(annotator_name).name        

    def create_job_dict(self, prompt, generation_config, idx=None):
        return {
            "model": self.annotator_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
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
            "metadata": { 
                "request_idx" : idx,
            } if idx is not None else {}
        }

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
            job = self.create_job_dict(data.user_prompts[idx], generation_config, idx)
            jobs.append(job)
        
        with open(jobs_file, "w") as f:
            for job in jobs:
                f.write(json.dumps(job) + "\n")
        print(f"Jobs file {jobs_file} has been created/overwritten.")
        
        return jobs
        
    def annotate(self, data, generation_config):
        n = len(data.user_prompts)

        # Create jobs.json for parallel request processing
        jobpath = f"datasets/temp/{data.data_path.replace('/', '_')}"
        os.makedirs(jobpath, exist_ok=True)
        jobs_file = f"{jobpath}/jobs.json"

        # Check if the jobs file already exists
        if os.path.exists(jobs_file):
            user_input = input(f"File {jobs_file} already exists. Do you want to override it? (y/N): ")
            if user_input.lower() != 'y':
                # Load existing jobs from file
                with open(jobs_file, "r") as f:
                    jobs = [json.loads(line) for line in f]
                print(f"Using existing jobs from {jobs_file}")
                print(f"Number of jobs: {len(jobs)}")
                print("Example job:")
                print(json.dumps(jobs[0], indent=2))
            else:
                # Create new jobs and write to file
                jobs = self._create_and_write_jobs(n, data, generation_config, jobs_file)
        else:
            # Create new jobs and write to file
            jobs = self._create_and_write_jobs(n, data, generation_config, jobs_file)

        print(f"Parallel processing starting, logging to {jobpath}/output.log")
        # Run batch processing
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=jobs_file,
                save_filepath=f"{jobpath}/output.jsonl",
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_requests_per_minute=float(self.config.max_requests_per_minute),
                max_tokens_per_minute=float(self.config.max_tokens_per_minute),
                token_encoding_name=self.encoder_name,
                max_attempts=5,
                logging_level=20,
                resume=self.config.resume,
                log_filepath=f"{jobpath}/output.log"
            )
        )
        print(f"Parallel processing complete. Check {jobpath}/output.log for details.")

        # Load file that was created
        outputs = {}
        with open(f"{jobpath}/output.jsonl", 'r') as f:
            for line in f:
                l = json.loads(line)
                outputs[l[2]['request_idx']] = l[1]['choices'][0]['message']['content']
        print(f"Number of outputs: {len(outputs)}")
        data.annotations = [outputs[i] for i in range(n)]