from typing import Union
from pathlib import Path
from PIL import Image


class Razrez:
    def __init__(self,
                 pre_pipe,
                 seg_pipe,
                 post_pipe):
        self.preprocessing_pipeline = pre_pipe
        self.segmentation_pipeline = seg_pipe
        self.postprocessing_pipeline = post_pipe

    def __call__(self, image: Union[str, Path, Image])->:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image)

