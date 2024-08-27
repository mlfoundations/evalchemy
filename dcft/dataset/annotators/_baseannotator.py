class BaseAnnotator:
    def __init__(self, annotator_name, **kwargs):
        self.annotator_name = annotator_name
        self.generation_args = kwargs

    def annotate(self, data):
        raise NotImplementedError
