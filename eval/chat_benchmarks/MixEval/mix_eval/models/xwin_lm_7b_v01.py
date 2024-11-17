import torch

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model


@register_model("xwin_lm_7b_v01")
class XWin_LM_7B_V01(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "Xwin-LM/Xwin-LM-7B-V0.1"
        self.attn_implementation = None  # If use default, set to None

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
        }
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.openended_max_new_tokens = 512
        self.gen_kwargs = {
            "temperature": 0.7,
        }

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

    def apply_chat_template(self, messages):
        prompt = ""
        if messages[0]["role"] == "system":
            prompt += f"""{messages[0]['content']}"""
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                prompt += f"""USER: {message['content']} """
            elif message["role"] == "assistant":
                prompt += f"""ASSISTANT: {message['content']}</s>"""

            if idx == len(messages) - 1:
                assert message["role"] == "user", "The last message must be from the user."
                prompt += f"""ASSISTANT:"""
        return prompt
