import os
import time
from dotenv import load_dotenv
import random

import google.generativeai as genai

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("gemini_10_pro")
class Gemini_10_Pro(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "gemini-1.0-pro-001"
        self.get_user_message = lambda prompt: {"role": "user", "parts": [prompt]}
        self.get_model_message = lambda response: {"role": "model", "parts": [response]}

        load_dotenv()
        genai.configure(api_key=os.getenv("k_g"))
        self.model = genai.GenerativeModel(self.model_name)

        self.safety_settings = {
            "harm_category_harassment": "block_none",
            "harm_category_hate_speech": "block_none",
            "harm_category_sexually_explicit": "block_none",
            "harm_category_dangerous_content": "block_none",
        }

    def _decode(self, inputs):
        completion = self.model.generate_content(
            inputs,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=self.MAX_NEW_TOKENS,
            ),
            safety_settings=self.safety_settings,
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.text

    def decode(self, inputs):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._decode(inputs)
                return response_content
            except Exception as e:
                if "quick accessor" in str(e) or "block" in str(e):
                    print("Content blocked, retrying ...")
                    blocked += 1
                    if blocked > 10:
                        print(
                            "Blocked for too many times, using 'Response not available "
                            "due to content restrictions.' as response, exiting..."
                        )
                        return "Response not available due to content restrictions."
                elif "quota" in str(e).lower() or "limit" in str(e).lower():
                    exponential_base = 2
                    delay *= exponential_base * (1 + random.random())
                    print(f"Error, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                    print(e)
                    time.sleep(delay)
                    continue
                else:
                    print(f"Error in decode, retrying...")
                    print(e)
                    time.sleep(10)
                    continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return "Error"
