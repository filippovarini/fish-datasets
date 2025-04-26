## The dataset can't be downloaded programmatically.
## Please download it from the following link:
## https://data.csiro.au/collection/64913
## 
## Once Downloaded, put the .zip file in the folder fish-datasets/data/raw/torsi (create the folder if not exists)

from pathlib import Path

import matplotlib.pyplot as plt

from aggregation_of_final_dataset.settings import Settings
from data_preview.utils import (
    extract_downloaded_file, 
    build_and_visualize_supervision_dataset_from_coco_dataset,
    save_sample_image
    )

settings = Settings()


DATASET_SHORTNAME = "torsi"
download_path = settings.raw_dir / DATASET_SHORTNAME
annotations_path = download_path / "data" / "instances.json"
images_path = download_path / "data"
zip_path = download_path / "Tasmanian_Orange_Roughy_Stereo_Image_Machine_Learning_Dataset-QEzDvqEq-.zip"



def download_data():
    how_to_download_the_data = """
    The dataset can't be downloaded programmatically.
    Please download it from the following link:
    https://data.csiro.au/collection/64913

    Once Downloaded, put the .zip file in the folder data/raw/torsi (create the folder if not exists)
    """

    print(f"Checking if data is already downloaded in {download_path}")
    if Path(download_path).exists():
        print("Data already downloaded. Extracting...")
        extract_downloaded_file(zip_path, extract_to=download_path)
        return
    else:
        raise NotImplementedError(how_to_download_the_data)

def main():
    # Download the data
    download_data()
    
    # The dataset is already in COCO format
    image_sample = build_and_visualize_supervision_dataset_from_coco_dataset(
        images_path, 
        annotations_path)
    save_sample_image(DATASET_SHORTNAME, image_sample)
    
    
    
if __name__ == "__main__":
    main()
