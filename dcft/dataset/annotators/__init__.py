from dcft.dataset.annotators.gpt import GPTAnnotator
from dcft.dataset.annotators.sambanova import SambaNovaAnnotator

ANNOTATOR_MAP = {
    "gpt-4o": GPTAnnotator,
    "gpt-4o-2024-05-13": GPTAnnotator,
    "gpt-4o-2024-08-06": GPTAnnotator,
    "chatgpt-4o-latest": GPTAnnotator,
    "gpt-4o-mini": GPTAnnotator,
    "gpt-4o-mini-2024-07-18": GPTAnnotator,
    "gpt-4-turbo": GPTAnnotator,
    "gpt-4-turbo-2024-04-09": GPTAnnotator,
    "gpt-4-turbo-preview": GPTAnnotator,
    "gpt-4-0125-preview": GPTAnnotator,
    "gpt-4-1106-preview": GPTAnnotator,
    "gpt-4": GPTAnnotator,
    "gpt-4-0613": GPTAnnotator,
    "gpt-4-0314": GPTAnnotator,
    "llama3-405b": SambaNovaAnnotator,
}


class AnnotatorConfig:
    def __init__(self, args):
        self.annotator_name = args.annotator

        if is_gpt_annotator(args.annotator):
            self.max_requests_per_minute = args.max_requests_per_minute
            self.max_tokens_per_minute = args.max_tokens_per_minute


def is_gpt_annotator(annotator_name):
    return ANNOTATOR_MAP[annotator_name] is GPTAnnotator


def get_annotator(annotator_name, annotator_config, **kwargs):
    return ANNOTATOR_MAP[annotator_name](annotator_name, annotator_config, **kwargs)
