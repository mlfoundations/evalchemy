import torch
import json

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("gemma_2_27b_instruct")
class Gemma_2_27B_Instruct(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "google/gemma-2-27b-it"
        self.attn_implementation = "eager"  # If use default, set to None
        self.model_dtype = torch.bfloat16

        self.SYSTEM_MESSAGE = None  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens
