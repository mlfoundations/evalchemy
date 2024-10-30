import os
import time
from dotenv import load_dotenv
import random

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud.aiplatform_v1beta1.types import Part
from proto import STRING

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model


@register_model("gemini_15_pro")
class Gemini_15_Pro(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = "gemini-1.5-pro"

        def get_user_message(prompt):
            part = Part()
            part.text = prompt
            return {"role": "user", "parts": [part]}

        def get_model_message(response):
            part = Part()
            part.text = response
            return {"role": "model", "parts": [part]}

        self.get_user_message = get_user_message
        self.get_model_message = get_model_message

        load_dotenv()
        project_id = os.getenv("id_g")
        location = "us-central1"
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(self.model_name)

        self.safety_config = {
            vertexai.generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
            vertexai.generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
            vertexai.generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
            vertexai.generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

    def _decode(self, inputs):
        response = self.model.generate_content(
            inputs,
            generation_config=GenerationConfig(
                candidate_count=1,
                max_output_tokens=self.MAX_NEW_TOKENS,
            ),
            safety_settings=self.safety_config,
            stream=False,
        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return response.text

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
