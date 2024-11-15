import os
from dotenv import load_dotenv

from client.snova_client import SnovaClient
from httpx import Timeout

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("llama_3_405b")
class Llama_3_405B(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "llama3-405b"

        load_dotenv()
        self.client = SnovaClient(key=os.getenv("k_snova"))
