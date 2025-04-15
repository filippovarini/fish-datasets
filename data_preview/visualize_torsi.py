from pathlib import Path

import matplotlib.pyplot as plt

from aggregation_of_final_dataset.settings import Settings
from data_preview.utils import extract_downloaded_file, build_and_visualize_supervision_dataset_from_coco_dataset

settings = Settings()


DATASET_SHORTNAME = "torsi"


def download_data(download_path: str):
    how_to_download_the_data = """
    The dataset can't be downloaded programmatically.
    Please download it from the following link:
    https://data.csiro.au/collection/64913
    """
    print(f"Checking if data is already downloaded in {download_path}")
    if Path(download_path).exists():
        print("Data already downloaded")
        return
    else:
        raise NotImplementedError(how_to_download_the_data)

def main():
    # Download the data
    download_path = settings.raw_dir / DATASET_SHORTNAME
    download_data(download_path=download_path)
    extract_downloaded_file(download_path=download_path / "torsi.zip", extract_to=download_path)
    
    # The dataset is already in COCO format
    annotations_path = download_path / "data" / "instances.json"
    images_path = download_path / "data"
    
    random_sample_image = build_and_visualize_supervision_dataset_from_coco_dataset(
        images_dir=images_path,
        annotations_path=annotations_path,
    )
    
    image_name = f"{DATASET_SHORTNAME}_sample_image.png"
    plt.imsave(image_name, random_sample_image)
    
    
    
if __name__ == "__main__":
    main()
