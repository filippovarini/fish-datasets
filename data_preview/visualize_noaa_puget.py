import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import json

from data_preview.utils import (
    visualize_supervision_dataset,
    download_file,
    extract_downloaded_file,
    CompressionType,
)


DATASET_SHORTNAME = "noaa_puget"
DATA_DIR = Path("/mnt/data/dev/fish-datasets/data/raw") / DATASET_SHORTNAME
ANNOTATIONS_PATH = DATA_DIR / "noaa_estuary_fish-2023.08.19.json"
IMAGES_PATH = DATA_DIR / "JPEGImages"


def clean_annotations(annotations_path: Path):
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    cleaned_annotations = []

    print(f"Number of annotations: {len(annotations['annotations'])}")

    for i, annotation in enumerate(annotations["annotations"]):
        if "bbox" not in annotation or len(annotation["bbox"]) == 0:
            print(f"No bbox found for {annotation['image_id']}")
        else:
            cleaned_annotations.append(annotation)

    annotations["annotations"] = cleaned_annotations

    with open(annotations_path, "w") as f:
        print(f"Number of annotations: {len(annotations['annotations'])}")
        json.dump(annotations, f)


def extract_example_image(
    images_path: Path, annotations_path: Path, dataset_shortname: str
):
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_path),
        annotations_path=str(annotations_path),
    )

    image_example = visualize_supervision_dataset(dataset)
    plt.imsave(f"{dataset_shortname}_sample_image.png", image_example)


def download_data(data_dir: Path):
    data_dir.mkdir(exist_ok=True, parents=True)

    data_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-images.zip"
    annotations_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-annotations-2023.08.19.zip"

    data_path_zip = data_dir / "images.zip"
    annotations_path_zip = data_dir / "annotations.zip"

    print("Extracting data...")
    download_file(data_url, data_path_zip)
    download_file(annotations_url, annotations_path_zip)

    extract_downloaded_file(data_path_zip, data_dir, CompressionType.ZIP)
    extract_downloaded_file(annotations_path_zip, data_dir, CompressionType.ZIP)


def main():
    download_data(DATA_DIR)

    clean_annotations(ANNOTATIONS_PATH)

    extract_example_image(IMAGES_PATH, ANNOTATIONS_PATH, DATASET_SHORTNAME)


# if __name__ == "__main__":
#     main()
