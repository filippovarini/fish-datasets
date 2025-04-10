from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    raw_dir: Path = Path("/mnt/data/dev/fish-datasets/data/raw")
    processed_dir: Path = Path("/mnt/data/dev/fish-datasets/data/final")
    
    train_dataset_suffix: str = "_train"
    val_dataset_suffix: str = "_val"
    images_folder_name: str = "JPEGImages"

    # We only use one category for the fish
    coco_category_id: int = 1
    coco_categories = [{"id": coco_category_id, "name": "fish"}]
    

