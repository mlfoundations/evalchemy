import torch
from transformers import AutoTokenizer

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("mpt_7b_chat")
class MPT_7B_Chat(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "mosaicml/mpt-7b-chat"
        self.attn_implementation = None  # If use default, set to None
        self.model_dtype = torch.bfloat16
        self.trust_remote_code = True
        self.use_fast_tokenizer = True
        self.openended_max_new_tokens = 512

        # system message from https://huggingface.co/mosaicml/mpt-7b-chat/discussions/40#659f8fb121c219062cf3c3ef
        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "You are Assistant. "
            "You were made to answer questions and be helpful.\n"
            "- You follow instructions\n"
            "- You are polite\n"
            "- You are helpful\n"
            "- You are friendly",
        }  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.gen_kwargs = {
            "do_sample": True,
        }

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_seq_len
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.model_max_len,
            padding_side=self.padding_side,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,
            revision="ed874721",
        )
        return tokenizer
