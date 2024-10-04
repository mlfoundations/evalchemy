import os
import time
from dotenv import load_dotenv
import random

import anthropic
from httpx import Timeout
from anthropic._exceptions import RateLimitError

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("claude_3_opus")
class Claude_3_Opus(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.FIX_INTERVAL_SECOND = 1

        self.model_name = "claude-3-opus-20240229"

        load_dotenv()
        self.client = anthropic.Anthropic(api_key=os.getenv("k_ant"), timeout=Timeout(timeout=20.0, connect=5.0))

    def _decode(self, inputs):
        completion = self.client.messages.create(model=self.model_name, max_tokens=self.MAX_NEW_TOKENS, messages=inputs)
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.content[0].text

    def decode(self, inputs):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                return response_content
            except RateLimitError as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                print(e)
                time.sleep(delay)
                continue
            except Exception as e:
                print(f"Error in decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"
