from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("mistral_7b_instruct_v02")
class Mistral_7B_Instruct_V02(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.attn_implementation = "flash_attention_2"  # If use default, set to None  # If use default, set to None

        self.SYSTEM_MESSAGE = None  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.gen_kwargs = {
            "do_sample": True,
        }

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens
