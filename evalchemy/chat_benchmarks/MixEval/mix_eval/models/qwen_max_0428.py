import os
import time
from dotenv import load_dotenv
import random
import subprocess

from httpx import Timeout
from http import HTTPStatus
from dashscope import Generation

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("qwen_max_0428")
class Qwen_Max_0428(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "qwen-max-0428"

    def _decode(self, inputs):
        if inputs[0]["role"] != "system":
            inputs = [
                {"role": "system", "content": "You are a helpful assistant."},
            ] + inputs
        completion = Generation.call(
            model=self.model_name, max_tokens=self.MAX_NEW_TOKENS, messages=inputs, result_format="message"
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion

    def decode(self, inputs):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                completion = self._decode(inputs)
                if completion.status_code == HTTPStatus.OK:
                    return completion.output.choices[0].message.content
                else:
                    raise Exception(completion)
            except Exception as e:
                if "rate" in str(e).lower():
                    exponential_base = 2
                    delay *= exponential_base * (1 + random.random())
                    print(f"Rate limit error, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                    print(e)
                    time.sleep(delay)
                    continue
                elif "Output data may contain inappropriate content." in str(e):
                    print("Content blocked, retrying ...")
                    blocked += 1
                    if blocked > 10:
                        print(
                            "Blocked for too many times, using 'Response not available "
                            "due to content restrictions.' as response, exiting..."
                        )
                        return "Response not available due to content restrictions."
                    continue
                else:
                    print(f"Error in decode, retrying...")
                    print(e)
                    time.sleep(5)
                    continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"
