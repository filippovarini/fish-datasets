from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    base_dir: Path = Path("/mnt/data/dev/fish-datasets/final_dataset")
    train_dataset_suffix: str = "_train"
    val_dataset_suffix: str = "_val"
    images_folder_name: str = "JPEGImages"

    # We only use one category for the fish
    coco_category_id: int = 1
    coco_categories = [{"id": coco_category_id, "name": "fish"}]
    

