from pathlib import Path

import gdown


DATASET_SHORTNAME = "vmat"
DATA_URL = "https://drive.google.com/drive/folders/18fknmUjD4aq3-Qktn-rLVIaNEEujV3wK"

def download_gdrive_folder(output_path: Path):
    """Download an entire folder from Google Drive using gdown."""
    
    print(f"Downloading folder from Google Drive (ID: {DATA_URL}) to {output_path}")
    
    try:
        # Use gdown's folder download capability
        gdown.download_folder(url=DATA_URL, output=str(output_path), quiet=False)
        print(f"Downloaded folder to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading folder from Google Drive: {e}")
        return False


def main():
    output_path = Path("~/data") / DATASET_SHORTNAME
    output_path.mkdir(exist_ok=True, parents=True)
    download_gdrive_folder(output_path)

# if __name__ == "__main__":
#     main()