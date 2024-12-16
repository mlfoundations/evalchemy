from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model


@register_model("mixtral_8_7b")
class Mixtral_8_7B(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "mistralai/Mixtral-8x7B-v0.1"
        self.attn_implementation = "flash_attention_2"  # If use default, set to None  # If use default, set to None

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
