import os
import shutil

from aggregation_of_final_dataset.settings import Settings
from data_preview.visualize_zebrafish import (
    DATASET_SHORTNAME,
    download_data,
    clean_annotations_and_get_df,
    dataframe_to_coco,
)
import tqdm


settings = Settings()


def main():
    # 1. RAW
    # Kagglehub downloads by default to  ~/.cache/kagglehub/
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(raw_download_path)
    download_data(raw_download_path)

    # 2. PROCESSING
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    # Clean and process annotations
    data_df, raw_images_path = clean_annotations_and_get_df(raw_download_path)
    dataframe_to_coco(data_df, coco_annotations_path)
    
    # Copy all raw images in coco images path
    images_generator = raw_images_path.glob("*.png")
    total_images = len(list(images_generator))
    for image_path in tqdm.tqdm(images_generator, total=total_images):
        shutil.copy2(image_path, coco_images_path / image_path.name)


if __name__ == "__main__":
    main()
