import torch
from transformers import AutoTokenizer

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("mpt_7b_instruct")
class MPT_7B_Instruct(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "mosaicml/mpt-7b-instruct"
        self.attn_implementation = None  # If use default, set to None
        self.model_dtype = torch.bfloat16
        self.trust_remote_code = True
        self.use_fast_tokenizer = True
        self.openended_max_new_tokens = 512

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n",
        }
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

    def apply_chat_template(self, messages):
        prompt = ""
        if messages[0]["role"] == "system":
            prompt += f"""{messages[0]['content']}"""
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                prompt += f"""### Instruction:\n{message['content']}\n"""
            elif message["role"] == "assistant":
                prompt += f"""### Response:\n{message['content']}\n"""

            if idx == len(messages) - 1:
                assert message["role"] == "user", "The last message must be from the user."
                prompt += f"""### Response:\n"""
        return prompt
