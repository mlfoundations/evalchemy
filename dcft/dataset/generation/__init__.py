import json
from typing import Optional


class GenerationConfig:
    def __init__(
        self,
        temperature: float,
        top_p: float,
        seed: int,
        max_tokens: int,
        stop: Optional[str],
        frequency_penalty: float,
        logit_bias: Optional[str],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        n: int,
        presence_penalty: float,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.stop = stop.split(",") if stop is not None else None
        self.frequency_penalty = frequency_penalty
        self.logit_bias = json.loads(logit_bias) if logit_bias is not None else None
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.n = n
        self.presence_penalty = presence_penalty
