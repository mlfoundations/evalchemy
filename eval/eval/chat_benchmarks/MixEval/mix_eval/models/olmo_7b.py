# todo you have to `pip install ai2-olmo` and `import hf_olmo` to run olmo,
# which may be conflict with the default setup.
# see https://huggingface.co/allenai/OLMo-7B

# import hf_olmo
import torch
from transformers import AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
from transformers import AutoModelForCausalLM

from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model


@register_model("olmo_7b")
class OLMo_7B(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "allenai/OLMo-7B"
        self.attn_implementation = None  # If use default, set to None
        self.trust_remote_code = False
        self.use_fast_tokenizer = True

        self.gen_kwargs = {"do_sample": True, "top_k": 50, "top_p": 0.95}

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_sequence_length
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens

    def build_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.model_dtype,
            trust_remote_code=self.trust_remote_code,
        ).eval()

        model.tie_weights()
        device_map = infer_auto_device_map(model)
        model = dispatch_model(
            model,
            device_map=device_map,
        )

        return model
