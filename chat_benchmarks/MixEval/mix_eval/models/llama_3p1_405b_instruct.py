import os
from dotenv import load_dotenv

from client.snova_client import SnovaClient
from httpx import Timeout

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("llama_3p1_405b_instruct")
class Llama_3p1_405B_Instruct(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "Meta-Llama-3.1-405B-Instruct"

        load_dotenv()
        self.client = SnovaClient(key=os.getenv("k_snova"))
