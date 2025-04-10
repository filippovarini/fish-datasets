from data_preview.visualise_noaa_puget import download_data, DATASET_SHORTNAME
from aggregation_of_final_dataset.utils import compress_annotations_to_single_category
from settings import Settings

settings = Settings()


def main():
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # Create COCO Dataset and store in intermediate directory
    # We compress all annotations into a single category: Fish
    raw_annotations_path = raw_download_path / "noaa_estuary_fish-2023.08.19.json"
    categories_to_keep = ["fish"]
    compressed_annotations_path = settings.intermediate_dir / "noaa_puget_compressed_annotations.json"
    compressed_annotations_path = compress_annotations_to_single_category(
        raw_annotations_path, categories_to_keep, compressed_annotations_path
    )

if __name__ == "__main__":
    main()
