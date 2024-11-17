import torch
from fastchat.model import load_model, add_model_args

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("vicuna_7b_v13")
class Vicuna_7B_V13(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "lmsys/vicuna-7b-v1.3"
        self.attn_implementation = "flash_attention_2"  # If use default, set to None  # If use default, set to None
        self.openended_max_new_tokens = 512

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        }  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}
        CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'system' %}{{ message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}{% endfor %}"""

        self.gen_kwargs = {"do_sample": True, "temperature": 0.7, "repetition_penalty": 1.0, "top_p": 1, "top_k": 50}
        self.model, self.tokenizer = self.load_vicuna_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer.chat_template = CHAT_TEMPLATE
        self.tokenizer.model_max_length = self.model_max_len
        self.tokenizer.padding_side = self.padding_side

        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

    def load_vicuna_model(self):
        return load_model(
            self.model_name,
            num_gpus=torch.cuda.device_count(),
            max_gpu_memory=self.args.max_gpu_memory,
        )
