from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    base_dir: Path = Path("/mnt/data/dev/fish-datasets/final_dataset")
    train_dataset_suffix: str = "_train"
    val_dataset_suffix: str = "_val"

