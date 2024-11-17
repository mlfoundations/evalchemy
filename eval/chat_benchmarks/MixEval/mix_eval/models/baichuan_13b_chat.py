import torch

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("baichuan_13b_chat")
class Baichuan_13B_Chat(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "baichuan-inc/Baichuan-13B-Chat"
        self.attn_implementation = None  # If use default, set to None
        self.trust_remote_code = True
        self.model_dtype = torch.float16

        self.SYSTEM_MESSAGE = None
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.model_max_length
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

    def apply_chat_template(self, messages):
        prompt = ""
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                prompt += f"""<reserved_102>{message['content']}"""
            elif message["role"] == "assistant":
                prompt += f"""<reserved_103>{message['content']}"""

            if idx == len(messages) - 1:
                assert message["role"] == "user", "The last message must be from the user."
                prompt += f"""<reserved_102>"""
        return prompt
