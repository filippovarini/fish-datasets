from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from coco_lib.bases import Annotation, Dataset


@dataclass_json
@dataclass
class ImageCaptioningAnnotation(Annotation):
    id: int
    image_id: int
    caption: str


@dataclass_json
@dataclass
class ImageCaptioningDataset(Dataset):
    annotations: List[ImageCaptioningAnnotation]