import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model
from mix_eval.utils.common_utils import get_gpu_memory


@register_model("baichuan2_13b_chat")
class Baichuan2_13B_Chat(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "baichuan-inc/Baichuan2-13B-Chat"
        self.attn_implementation = None  # If use default, set to None
        self.trust_remote_code = True
        self.model_dtype = torch.bfloat16

        self.SYSTEM_MESSAGE = None
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model = self.build_model()
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_name, revision="v2.0")
        self.model_max_len = self.model.config.model_max_length
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(self.model_max_len, self.max_input_length) - self.openended_max_new_tokens

    def build_model(self):
        num_gpus = torch.cuda.device_count()
        kwargs = {}
        kwargs["device_map"] = "auto"
        if self.args.max_gpu_memory is None:
            kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
            available_gpu_memory = get_gpu_memory(num_gpus)
            kwargs["max_memory"] = {i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(num_gpus)}
        else:
            kwargs["max_memory"] = {i: self.args.max_gpu_memory for i in range(num_gpus)}

        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.model_dtype,
            trust_remote_code=self.trust_remote_code,
            revision="v2.0",
            **kwargs,
        ).eval()
        return model

    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.model_max_len,
            padding_side=self.padding_side,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,
            revision="v2.0",
        )
        return tokenizer

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
