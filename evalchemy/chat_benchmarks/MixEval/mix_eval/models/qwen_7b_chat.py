from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("qwen_7b_chat")
class Qwen_7B_Chat(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "Qwen/Qwen-7B-Chat"
        self.attn_implementation = None  # If use default, set to None
        self.trust_remote_code = True

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token_id = 151643
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

        self.gen_kwargs = {
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def apply_chat_template(self, messages):
        prompt = ""
        if messages[0]["role"] == "system":
            prompt += f"""<|im_start|>system\n{messages[0]['content']}<|im_end|>"""
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                prompt += f"""\n<|im_start|>user\n{message['content']}<|im_end|>"""
            elif message["role"] == "assistant":
                prompt += f"""\n<|im_start|>assistant\n{message['content']}<|im_end|>"""

            if idx == len(messages) - 1:
                assert message["role"] == "user", "The last message must be from the user."
                prompt += f"""\n<|im_start|>assistant\n"""
        return prompt
