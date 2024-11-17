import torch

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("internlm2_chat_7b")
class InternLM2_Chat_7B(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "internlm/internlm2-chat-7b"
        self.attn_implementation = None  # If use default, set to None
        self.trust_remote_code = True
        self.model_dtype = torch.float16

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "You are an AI assistant whose name is InternLM (书生·浦语).\n"
            "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
            "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
        }
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens
