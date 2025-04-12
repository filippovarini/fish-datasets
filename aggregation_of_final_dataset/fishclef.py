import json

from data_preview.visualize_fishclef import (
    DATASET_SHORTNAME,
    download_data,
    convert_annotations,
    extract_frames_from_videos,
    merge_coco_datasets_into_single_dataset,
)
from aggregation_of_final_dataset.settings import Settings


settings = Settings()


def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESSING
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    convert_annotations(raw_download_path, processing_dir)
    
    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    all_annotation_paths = list(processing_dir.rglob("*.json"))
    print(f"Merging {len(all_annotation_paths)} annotation files")
    merge_coco_datasets_into_single_dataset(all_annotation_paths, coco_annotations_path)
    
    with open(coco_annotations_path, "r") as f:
        coco_annotations = json.load(f)
    
    extract_frames_from_videos(raw_download_path, coco_images_path, coco_annotations)


if __name__ == "__main__":
    main()
