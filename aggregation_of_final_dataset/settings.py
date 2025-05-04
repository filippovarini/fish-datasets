from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    raw_dir: Path = Path("/Users/filippovarini/Desktop/Development/fish-datasets/data/raw")
    processed_dir: Path = Path("/Users/filippovarini/Desktop/Development/fish-datasets/data/final")
    intermediate_dir: Path = Path("/Users/filippovarini/Desktop/Development/fish-datasets/data/processing")
    
    train_dataset_suffix: str = "_train"
    val_dataset_suffix: str = "_val"
    images_folder_name: str = "JPEGImages"

    # We only use one category for the fish
    coco_category_id: int = 1
    coco_categories = [{"id": coco_category_id, "name": "fish"}]
    coco_file_name: str = "annotations_coco.json"
    
    # AI
    train_val_split_ratio: float = 0.2
    random_state: int = 42
    
    

