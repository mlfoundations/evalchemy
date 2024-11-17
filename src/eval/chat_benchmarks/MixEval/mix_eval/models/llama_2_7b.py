from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model


@register_model("llama_2_7b")
class LLAMA_2_7B(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "meta-llama/Llama-2-7b-hf"
        self.attn_implementation = (
            None  # replaced "flash_attention_2"  # If use default, set to None  # If use default, set to None
        )

        self.model = self.build_model().bfloat16()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
