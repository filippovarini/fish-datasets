from data_preview.visualize_vmat import DATASET_SHORTNAME, download_gdrive_folder
from aggregation_of_final_dataset.settings import Settings


settings = Settings()


def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_gdrive_folder(raw_download_path)


if __name__ == "__main__":
    main()
