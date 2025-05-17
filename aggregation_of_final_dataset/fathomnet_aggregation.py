from data_preview.visualize_fathomnet import download_data, DATASET_SHORTNAME
from settings import Settings


settings = Settings()
def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)
    

if __name__ == "__main__":
    main()
