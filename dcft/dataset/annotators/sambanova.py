import os
import time
import json
import requests
from tqdm import tqdm
from dcft.dataset.annotators._baseannotator import BaseAnnotator

class SambaNovaAnnotator(BaseAnnotator):
    def __init__(self, annotator_name, annotator_config, **kwargs):
        super().__init__(annotator_name, annotator_config, **kwargs)
        self.NUM_SECONDS_TO_SLEEP = 30
        self.key = os.getenv("SAMBANOVA_API_KEY")
        assert self.key is not None
        self.url = "https://fast-api.snova.ai/v1/chat/completions"
        self.headers = {"Authorization": f"Basic {self.key}", "Content-Type": "application/json"}


    def annotate(self, data, generation_config):
        n = len(data.user_prompts)
        for idx in tqdm(range(n)):
            messages = [
                {"role": "system", "content": data.system_prompts[idx]},
                {"role": "user", "content": data.user_prompts[idx]}
            ]
            payload = {
                "messages": messages, 
                "max_tokens": 800, 
                "stop": ["[INST", "[INST]", "[/INST]", "[/INST]"], 
                "model": "llama3-405b", 
                "stream": True, 
                "stream_options": 
                {"include_usage": True}
            }
            while True:
                post_response = requests.post(self.url, json=payload, headers=self.headers, stream=True)

                if post_response.status_code == 503 or post_response.status_code == 504 or post_response.status_code == 401 or post_response.status_code == 429:
                    print(post_response.content)
                    print(f"Attempt failed due to rate limit or gate timeout. Trying again...")
                    time.sleep(self.NUM_SECONDS_TO_SLEEP)
                    continue
                response_text = ""
                for line in post_response.iter_lines():
                    if line.startswith(b"data: "):
                        data_str = line.decode("utf-8")[6:]
                        try:
                            line_json = json.loads(data_str)
                            if "choices" in line_json and len(line_json['choices']) > 0 and "content" in line_json["choices"][0]["delta"]:
                                try:
                                    response_text += line_json["choices"][0]["delta"]["content"]
                                except:
                                    breakpoint()
                        except json.JSONDecodeError as e:
                            pass
                break
            data.annotations.append(response_text)
