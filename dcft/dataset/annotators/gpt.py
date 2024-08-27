
import os
import json
from tqdm import tqdm
from openai import OpenAI
from dcft.dataset.annotators._baseannotator import BaseAnnotator


class GPTAnnotator(BaseAnnotator):
    def __init__(self, annotator_name, **kwargs):
        super().__init__(annotator_name, **kwargs)
        self.client = OpenAI()

    def annotate(self, data):
        n = len(data.user_prompts)

        # Create jobs.json for batch processing
        jobs = []
        for idx in tqdm(range(n)):
            job = {
                "model": self.annotator_name,
                "messages": [
                    {"role": "user", "content": data.user_prompts[idx]}
                ],
                **self.generation_args
            }
            jobs.append(job)
        jobpath = f"datasets/temp/{data.data_path.replace('/', '_')}"
        os.makedirs(jobpath, exist_ok=True)
        with open(f"{jobpath}/jobs.json", "w") as f:
            for j in jobs:
                f.write(json.dumps(j)+'\n')

        # Run batch processing
        cmd = f"python dcft/utils/api_request_parallel_processor.py --requests_filepath {jobpath}/jobs.json --save_filepath {jobpath}/output.jsonl --max_requests_per_minute 1500 --max_tokens_per_minute 6250000 --token_encoding_name cl100k_base --max_attempts 5 --logging_level 20"
        os.system(cmd)

        # Load file that was created
        outputs = {}
        with open(f"{jobpath}/output.jsonl", 'r') as f:
            for line in f:
                l = json.loads(line)
                outputs[l[0]['messages'][0]['content']] = l[1]['choices'][0]['message']['content']
        for d in data.user_prompts:
            data.annotations.append(outputs[d])