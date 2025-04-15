from dataclasses import dataclass
from typing import List, Tuple

from dataclasses_json import dataclass_json

from coco_lib.bases import Annotation, Category, Dataset


@dataclass_json
@dataclass
class SegmentInfo:
    id: int
    category_id: int
    area: float
    bbox: Tuple[float, float, float, float]
    iscrowd: int


@dataclass_json
@dataclass
class PanopticSegmentationAnnotation(Annotation):
    image_id: int
    file_name: str
    segments_info: List[SegmentInfo]


@dataclass_json
@dataclass
class PanopticSegmentationCategory(Category):
    id: int
    name: str
    supercategory: str
    isthing: int
    color: Tuple[int, int, int]


@dataclass_json
@dataclass
class PanopticSegmentationDataset(Dataset):
    annotations: List[PanopticSegmentationAnnotation]
    categories: List[PanopticSegmentationCategory]