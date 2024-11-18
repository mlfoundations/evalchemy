import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model
from mix_eval.utils.common_utils import get_gpu_memory


@register_model("baichuan2_7b_chat")
class Baichuan2_7B_Chat(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "baichuan-inc/Baichuan2-7B-Chat"
        self.attn_implementation = None  # If use default, set to None
        self.trust_remote_code = True
        self.model_dtype = torch.float16

        self.SYSTEM_MESSAGE = None
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

    def apply_chat_template(self, messages):
        prompt = ""
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                prompt += f"""<reserved_106>{message['content']}"""
            elif message["role"] == "assistant":
                prompt += f"""<reserved_107>{message['content']}"""

            if idx == len(messages) - 1:
                assert message["role"] == "user", "The last message must be from the user."
                prompt += f"""<reserved_107>"""
        return prompt
