import os
from dotenv import load_dotenv
import time

from openai import OpenAI
from httpx import Timeout

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("openai_o1")
class OpenAI_o1(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "o1-preview-2024-09-12"

        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("k_oai"), timeout=Timeout(timeout=100.0, connect=20.0))

    def _decode(self, inputs):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "text"},
            max_completion_tokens=self.MAX_NEW_TOKENS,
            messages=inputs,
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.choices[0].message.content
