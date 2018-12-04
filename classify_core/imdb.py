import os

class IMDB(object):
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.annotations = self.load_annotations()

    def get_annotations(self):
        return self.annotations

    def load_annotations(self):
        assert os.path.exists(self.annotation_file), 'annotations not found at {}'.format(self.annotation_file)
        with open(self.annotation_file, 'r') as f:
            annotations = f.readlines()
        return annotations
