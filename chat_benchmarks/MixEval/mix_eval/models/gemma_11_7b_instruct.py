from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("gemma_11_7b_instruct")
class Gemma_11_7B_Instruct(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "google/gemma-1.1-7b-it"
        self.attn_implementation = "flash_attention_2"  # If use default, set to None  # If use default, set to None

        self.SYSTEM_MESSAGE = None  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = 2048
        self.max_input_length_openend = 2048
