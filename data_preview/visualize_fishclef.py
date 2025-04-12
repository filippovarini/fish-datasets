from pathlib import Path

from data_preview.utils import download_and_extract


DATASET_SHORTNAME = "fishclef"
DATA_URL = "https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1"


def download_data(output_path: Path):
    download_and_extract(output_path, DATA_URL, DATASET_SHORTNAME)


def main():
    output_path = Path("~/data") / DATASET_SHORTNAME
    download_data(output_path)


# if __name__ == "__main__":
#     main()
