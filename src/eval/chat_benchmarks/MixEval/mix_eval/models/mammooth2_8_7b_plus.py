from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("mammooth2_8_7b_plus")
class MAmmooTH2_8_7B_Plus(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "TIGER-Lab/MAmmoTH2-8x7B-Plus"
        self.attn_implementation = "flash_attention_2"  # If use default, set to None  # If use default, set to None

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "You are supposed to provide a solution to a given problem.\n\n",
        }

        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens
