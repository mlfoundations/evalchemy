import os
from dotenv import load_dotenv

from openai import OpenAI
from httpx import Timeout

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("gpt_4o_mini")
class GPT_4o_Mini(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "gpt-4o-mini"

        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("k_oai"), timeout=Timeout(timeout=20.0, connect=5.0))
