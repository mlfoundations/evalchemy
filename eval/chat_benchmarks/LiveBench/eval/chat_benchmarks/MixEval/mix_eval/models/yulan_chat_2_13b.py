import torch

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("yulan_chat_2_13b")
class Yulan_Chat_2_13B(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "yulan-team/YuLan-Chat-2-13b-fp16"
        self.attn_implementation = None  # If use default, set to None
        self.model_dtype = torch.bfloat16

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "The following is a conversation between a human and an AI assistant namely YuLan, developed by GSAI, Renmin University of China. The AI assistant gives helpful, detailed, and polite answers to the user's questions.",
        }
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

        self.gen_kwargs = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 64,
            "max_length": 8192,
            "pad_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def apply_chat_template(self, messages):
        prompt = ""
        if messages[0]["role"] == "system":
            prompt += f"""{messages[0]['content']}"""
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                prompt += f"""\n[|Human|]:{message['content']}"""
            elif message["role"] == "assistant":
                prompt += f"""\n[|AI|]:{message['content']}"""

            if idx == len(messages) - 1:
                assert message["role"] == "user", "The last message must be from the user."
                prompt += f"""\n[|AI|]:"""
        return prompt
