from pathlib import Path

from data_preview.utils import (
    download_and_extract
)


DATASET_SHORTNAME = "mit_river_herring"
DATA_DIR = Path("/mnt/data/dev/fish-datasets/data/raw") / DATASET_SHORTNAME

def download_data(data_dir: Path):
    if data_dir.exists():
        print(f"Data already exists: {data_dir}")
        return

    data_dir.mkdir(exist_ok=True, parents=True)

    data_url = "https://storage.googleapis.com/public-datasets-lila/mit-river-herring/mit_river_herring.zip"

    data_path_zip = data_dir / "images.zip"
    
    print("Extracting data...")
    download_and_extract(data_dir, data_url, DATASET_SHORTNAME)

def main():
    download_data(DATA_DIR)

if __name__ == "__main__":
    main()
    