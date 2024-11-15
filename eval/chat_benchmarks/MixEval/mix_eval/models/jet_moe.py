# refer to https://github.com/myshell-ai/JetMoE to install jetmoe
# pip install https://github.com/myshell-ai/JetMoE.git
# from jetmoe import JetMoEForCausalLM, JetMoEConfig, JetMoEForSequenceClassification

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification

from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model


@register_model("jet_moe")
class JetMoE(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "jetmoe/jetmoe-8b"
        self.attn_implementation = "eager"  # If use default, set to None
        self.model_dtype = torch.bfloat16
        self.trust_remote_code = True

        # self.gen_kwargs = {
        #     'num_return_sequences': 1,
        #     'no_repeat_ngram_size': 2
        # }

        AutoConfig.register("jetmoe", JetMoEConfig)
        AutoModelForCausalLM.register(JetMoEConfig, JetMoEForCausalLM)
        AutoModelForSequenceClassification.register(JetMoEConfig, JetMoEForSequenceClassification)
        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(self.model_max_len, self.max_input_length) - self.closeended_max_new_tokens
