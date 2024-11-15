from dcft.dataset.annotators.gpt import GPTAnnotator
from dcft.dataset.annotators.sambanova import SambaNovaAnnotator
from typing import Optional

# encourage people to pick versioned models for better reproducibility
ANNOTATOR_MAP = {
    "gpt-4o": GPTAnnotator,
    "gpt-4o-2024-05-13": GPTAnnotator,
    "gpt-4o-2024-08-06": GPTAnnotator,  # this is 50% cheaper than 5-13
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
    def __init__(
        self,
        annotator_name: str,
        resume: bool,
        max_requests_per_minute: Optional[int],
        max_tokens_per_minute: Optional[int],
        batch: bool,
        max_batch_size: Optional[int],
    ) -> None:
        self.annotator_name = annotator_name
        # only used by GPT annotators
        self.resume = resume
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.batch = batch
        self.max_batch_size = max_batch_size  # max size allowed 50k and 100MB files for each batch job


def is_gpt_annotator(annotator_name):
    return ANNOTATOR_MAP[annotator_name] is GPTAnnotator


def get_annotator(annotator_name, annotator_config, **kwargs):
    return ANNOTATOR_MAP[annotator_name](annotator_name, annotator_config, **kwargs)
