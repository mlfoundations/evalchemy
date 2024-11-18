from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model
from transformers import GenerationConfig


@register_model("deepseek_moe_16b_chat")
class Deepseek_MoE_16B_Chat(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "deepseek-ai/deepseek-moe-16b-chat"
        self.attn_implementation = None  # If use default, set to None
        self.trust_remote_code = True

        self.SYSTEM_MESSAGE = None  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens
