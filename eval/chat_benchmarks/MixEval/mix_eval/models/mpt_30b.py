import torch

from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model


@register_model("mpt_30b")
class MPT_30B(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "mosaicml/mpt-30b"
        self.attn_implementation = None  # If use default, set to None
        self.model_dtype = torch.bfloat16
        self.trust_remote_code = True
        self.use_fast_tokenizer = True

        self.gen_kwargs = {
            "do_sample": True,
        }

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_seq_len
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
