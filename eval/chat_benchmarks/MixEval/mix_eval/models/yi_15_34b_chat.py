from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("yi_15_34b_chat")
class Yi_15_34B_Chat(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "01-ai/Yi-1.5-34B-Chat"
        self.attn_implementation = None  # If use default, set to None

        self.SYSTEM_MESSAGE = None  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

        self.gen_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
        }
