import os
from dotenv import load_dotenv
import random
import time

from openai import OpenAI
from httpx import Timeout
from concurrent.futures import ThreadPoolExecutor
from openai._exceptions import RateLimitError

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("deepseek_v2")
class Deepseek_v2(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "deepseek-chat"

        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("d_sk"), timeout=Timeout(timeout=100.0, connect=20.0), base_url="https://api.deepseek.com"
        )

    def decode(self, inputs):
        delay = 1
        blocked = 0
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
                if "Content Exists Risk" in str(e):
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
                    time.sleep(1)
                    continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"
