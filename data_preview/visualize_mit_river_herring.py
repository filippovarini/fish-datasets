from pathlib import Path

import matplotlib.pyplot as plt

from data_preview.utils import (
    download_and_extract,
    build_and_visualize_supervision_dataset_from_coco_dataset
)


DATASET_SHORTNAME = "mit_river_herring"
DATA_DIR = Path("data/raw") / DATASET_SHORTNAME

def download_data(data_dir: Path):
    if data_dir.exists():
        print(f"Data already exists: {data_dir}")
        return

    data_dir.mkdir(exist_ok=True, parents=True)

    data_url = "https://storage.googleapis.com/public-datasets-lila/mit-river-herring/mit_river_herring.zip"

    print("Extracting data...")
    download_and_extract(data_dir, data_url, DATASET_SHORTNAME)
    
    annotations_path = data_dir / "mit_river_herring" / "mit_sea_grant_river_herring.json"
    images_path = data_dir / "mit_river_herring"
    
    return annotations_path, images_path


def main():
    annotations_path, images_path = download_data(DATA_DIR)
    
    image_example = build_and_visualize_supervision_dataset_from_coco_dataset(
        images_dir=images_path,
        annotations_path=annotations_path,
    )
    plt.imsave(f"data_preview/{DATASET_SHORTNAME}_sample_image.png", image_example)
    

if __name__ == "__main__":
    main()
    