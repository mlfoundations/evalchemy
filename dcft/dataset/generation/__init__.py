import json
from dcft.dataset.annotators import ANNOTATOR_MAP
from dcft.dataset.annotators.gpt import GPTAnnotator

class GenerationConfig:
    def __init__(self, args):
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seed = args.seed
        self.max_tokens = args.max_tokens
        self.stop = args.stop.split(',') if args.stop is not None else None

        if ANNOTATOR_MAP[args.annotator] is GPTAnnotator:
            self.frequency_penalty = args.frequency_penalty
            self.logit_bias = json.loads(args.logit_bias) if args.logit_bias is not None else None
            self.logprobs = args.logprobs
            self.top_logprobs = args.top_logprobs
            self.n = args.n
            self.presence_penalty = args.presence_penalty
