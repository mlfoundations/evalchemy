import os
import time
from dotenv import load_dotenv
import random

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from httpx import Timeout

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("mistral_nemo")
class Mistral_Nemo(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "open-mistral-nemo"

        load_dotenv()
        self.client = MistralClient(api_key=os.getenv("k_mis"), timeout=Timeout(timeout=120.0, connect=5.0))

    def _decode(self, inputs):
        inputs = [ChatMessage(role=message["role"], content=message["content"]) for message in inputs]
        completion = self.client.chat(model=self.model_name, max_tokens=self.MAX_NEW_TOKENS, messages=inputs)
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.choices[0].message.content

    def decode(self, inputs):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                return response_content
            except Exception as e:
                if "rate" in str(e).lower():
                    exponential_base = 2
                    delay *= exponential_base * (1 + random.random())
                    print(f"Rate limit error, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                    print(e)
                    time.sleep(delay)
                    continue
                else:
                    print(f"Error in decode, retrying...")
                    print(e)
                    time.sleep(5)
                    continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"
