import os
import time
from dotenv import load_dotenv
import random

import reka

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("reka_flash")
class Reka_Flash(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "reka-flash"
        self.get_user_message = lambda prompt: {"type": "human", "text": prompt}
        self.get_model_message = lambda response: {"type": "model", "text": response}

        load_dotenv()
        reka.API_KEY = os.getenv("k_reka")

    def _decode(self, inputs):
        current_turn = inputs[-1]
        conversation_history = inputs[:-1]
        completion = reka.chat(
            current_turn["text"],
            model_name=self.model_name,
            conversation_history=conversation_history,
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion["text"]

    def decode(self, inputs):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                return response_content
            except Exception as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(f"Error, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                print(e)
                time.sleep(delay)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"
