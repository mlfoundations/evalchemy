from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dcft.dataset.annotators import AnnotatorConfig
    from dcft.dataset.generation import GenerationConfig


class BaseAnnotator:
    def __init__(self, annotator_name: str, annotator_config: "AnnotatorConfig", **kwargs: Any) -> None:
        self.annotator_name = annotator_name
        self.config = annotator_config

    def annotate(self, data: Any, generation_config: "GenerationConfig", temp_dir: str) -> None:
        raise NotImplementedError
