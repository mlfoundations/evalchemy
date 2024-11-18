import torch

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("tigerbot_13b_chat_v1")
class TigerBot_13B_Chat_V1(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "TigerResearch/tigerbot-13b-chat-v1"
        self.attn_implementation = None  # If use default, set to None
        self.model_dtype = torch.bfloat16
        self.use_fast_tokenizer = True

        self.SYSTEM_MESSAGE = None  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

    def apply_chat_template(self, messages):
        prompt = ""
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                prompt += f"""\n\n### Instruction:\n{message['content']}"""
            elif message["role"] == "assistant":
                prompt += f"""\n\n### Response:\n{message['content']}"""

            if idx == len(messages) - 1:
                assert message["role"] == "user", "The last message must be from the user."
                prompt += f"""\n\n### Response:\n"""
        return prompt
