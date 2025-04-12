from data_preview.visualize_fishclef import (
    DATASET_SHORTNAME,
    download_data,
    convert_annotations,
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


if __name__ == "__main__":
    main()
