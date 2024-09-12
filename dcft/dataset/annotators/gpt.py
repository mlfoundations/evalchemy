import os
import json
from tqdm import tqdm
import tiktoken
from openai import OpenAI
from dcft.dataset.annotators._baseannotator import BaseAnnotator


class GPTAnnotator(BaseAnnotator):
    def __init__(self, annotator_name, annotator_config, **kwargs):
        super().__init__(annotator_name, annotator_config, **kwargs)
        self.client = OpenAI()
        self.encoder_name = tiktoken.encoding_for_model(annotator_name).name

    def create_job_dict(self, prompt, generation_config, idx=None):
        return {
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
            "metadata": (
                {
                    "request_idx": idx,
                }
                if idx is not None
                else {}
            ),
        }

    def annotate(self, data, generation_config):
        n = len(data.user_prompts)

        # Create jobs.json for batch processing
        jobs = []
        for idx in tqdm(range(n)):
            job = self.create_job_dict(data.user_prompts[idx], generation_config, idx)
            jobs.append(job)
        jobpath = f"datasets/temp/{data.data_path.replace('/', '_')}"
        os.makedirs(jobpath, exist_ok=True)
        with open(f"{jobpath}/jobs.json", "w") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")

        # Run batch processing
        cmd = f"python dcft/utils/api_request_parallel_processor.py \
            --requests_filepath {jobpath}/jobs.json \
            --save_filepath {jobpath}/output.jsonl \
            --max_requests_per_minute {self.config.max_requests_per_minute} \
            --max_tokens_per_minute {self.config.max_tokens_per_minute} \
            --token_encoding_name {self.encoder_name} \
            --max_attempts 5 \
            --logging_level 20"
        os.system(cmd)

        # Load file that was created
        outputs = {}
        with open(f"{jobpath}/output.jsonl", "r") as f:
            for line in f:
                l = json.loads(line)
                outputs[l[2]["request_idx"]] = l[1]["choices"][0]["message"]["content"]
        data.annotations = [outputs[i] for i in range(n)]
