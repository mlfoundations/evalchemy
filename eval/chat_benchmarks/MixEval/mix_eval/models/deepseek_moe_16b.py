from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model
from transformers import GenerationConfig


@register_model("deepseek_moe_16b")
class Deepseek_MoE_16B(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "deepseek-ai/deepseek-moe-16b-base"
        self.attn_implementation = None  # If use default, set to None
        self.trust_remote_code = True

        self.model = self.build_model()
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
