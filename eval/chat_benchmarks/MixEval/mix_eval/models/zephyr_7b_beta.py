import torch

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("zephyr_7b_beta")
class Zephyr_7B_Beta(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "HuggingFaceH4/zephyr-7b-beta"
        self.attn_implementation = "flash_attention_2"  # If use default, set to None  # If use default, set to None
        self.model_dtype = torch.bfloat16
        self.trust_remote_code = True

        # self.SYSTEM_MESSAGE = {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate."} # set to None if no system message
        self.SYSTEM_MESSAGE = None
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.gen_kwargs = {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens
