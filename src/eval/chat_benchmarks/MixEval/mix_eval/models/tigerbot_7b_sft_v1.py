import torch

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("tigerbot_7b_sft_v1")
class TigerBot_7B_SFT_V1(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "TigerResearch/tigerbot-7b-sft-v1"
        self.use_fast_tokenizer = True
        self.attn_implementation = None  # If use default, set to None
        self.model_dtype = torch.bfloat16
        self.trust_remote_code = True
        self.openended_max_new_tokens = 512

        self.SYSTEM_MESSAGE = None  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = 1000
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

        self.gen_kwargs = {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.8,
            "no_repeat_ngram_size": 4,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

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
