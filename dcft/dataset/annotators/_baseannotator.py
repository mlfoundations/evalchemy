class BaseAnnotator:
    def __init__(self, annotator_name, annotator_config, **kwargs):
        self.annotator_name = annotator_name
        self.config = annotator_config

    def annotate(self, data, generation_config):
        raise NotImplementedError
