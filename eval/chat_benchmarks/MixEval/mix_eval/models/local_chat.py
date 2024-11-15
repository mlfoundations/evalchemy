import torch
from transformers import AutoTokenizer

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("local_chat")
class LocalChatModel(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_path  # updates path to local model
        self.attn_implementation = "flash_attention_2"  # If use default, set to None  # If use default, set to None
        self.model_dtype = torch.bfloat16
        self.trust_remote_code = True

        if args.model_systemprompt:
            self.SYSTEM_MESSAGE = {"role": "system", "content": args.model_systemprompt}
        else:
            self.SYSTEM_MESSAGE = None
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.model_max_len,
            padding_side=self.padding_side,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
