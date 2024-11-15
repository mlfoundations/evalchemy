from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model


@register_model("gemma_7b")
class Gemma_7B(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "google/gemma-7b"
        self.attn_implementation = "flash_attention_2"  # If use default, set to None  # If use default, set to None

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()

        self.max_input_length_closeend = 2048
