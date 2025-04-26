from pathlib import Path
import os
import csv
import json
import shutil

import matplotlib.pyplot as plt

from data_preview.utils import (
    download_and_extract,
    CompressionType,
    build_and_visualize_supervision_dataset_from_coco_dataset,
)


DATASET_SHORTNAME = "deep_vision"
DATA_URL = (
    "https://ftp.nmdc.no/nmdc/IMR/MachineLearning/fishDatasetSimulationAlgorithm.zip"
)


def download_data(download_path: Path):
    download_and_extract(
        download_path, DATA_URL, DATASET_SHORTNAME, CompressionType.ZIP
    )


def csvs_to_coco(download_dir: Path, csv_files, images_path, output_json):
    """
    Converts multiple CSV files with annotations to a COCO-format JSON file.

    Args:
        csv_files (list of Path): List of paths to the CSV files.
        images_path (Path): Path to the images directory.
        output_json (Path): Path to save the output COCO JSON.
    """
    # Dictionaries for images and categories; list for annotations
    images = {}
    annotations = []
    categories = {}

    ann_id = 1  # Unique annotation id
    image_id = 1  # Unique image id

    # Process each CSV file
    for csv_file in csv_files:
        with csv_file.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 6:
                    continue
                rel_file_name, xmin, ymin, xmax, ymax, label = row

                # Remove any leading "/" from relative path if present
                rel_file_name = rel_file_name.lstrip("/")
                raw_image_path = download_dir / "fish_dataset" / rel_file_name
                file_name = raw_image_path.name

                # Copy image to images_path
                try:
                    shutil.copy2(raw_image_path, images_path / file_name)
                except Exception as e:
                    print(f"Error copying image {file_name}: {e}")
                    continue

                try:
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                except ValueError:
                    # Skip rows with invalid coordinate data
                    continue

                width = xmax - xmin
                height = ymax - ymin

                # Add image entry if not already added
                if file_name not in images:
                    images[file_name] = {
                        "id": image_id,
                        "file_name": file_name,
                        "width": None,  # Optionally, set the width if known
                        "height": None,  # Optionally, set the height if known
                    }
                    image_id += 1

                # Add category entry if not already added
                if label not in categories:
                    cat_id = len(categories) + 1  # unique category id
                    categories[label] = {
                        "id": cat_id,
                        "name": label,
                        "supercategory": label,  # or assign a default supercategory
                    }
                cat_id = categories[label]["id"]

                # Create annotation entry
                ann = {
                    "id": ann_id,
                    "image_id": images[file_name]["id"],
                    "category_id": cat_id,
                    "bbox": [
                        xmin,
                        ymin,
                        width,
                        height,
                    ],  # COCO format: [x, y, width, height]
                    "area": width * height,
                    "iscrowd": 0,
                }
                annotations.append(ann)
                ann_id += 1

    # Convert dictionaries to lists
    coco_images = list(images.values())
    coco_categories = list(categories.values())

    coco_dict = {
        "images": coco_images,
        "annotations": annotations,
        "categories": coco_categories,
    }

    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"COCO annotations saved to {output_json}")
    print(f"Number of images: {len(coco_images)}")
    print(f"Number of annotations: {len(annotations)}")

    return len(coco_images), len(annotations)


def create_coco_dataset(download_dir: Path, images_path: Path, annotations_path: Path):
    """Convert CSV annotations to COCO format."""
    csv_files = [
        download_dir / "fish_dataset/2017/train/source-train2017-annotations.csv",
        download_dir / "fish_dataset/2017/test/test_2017_annotations.csv",
        download_dir / "fish_dataset/2018/train/source-train2018-annotations.csv",
        download_dir / "fish_dataset/2018/test/test_2018_annotations.csv",
    ]

    json_annotations_path = annotations_path

    if json_annotations_path.exists():
        print(f"COCO dataset already exists: {json_annotations_path}")
    else:
        csvs_to_coco(download_dir, csv_files, images_path, json_annotations_path)

    return images_path, json_annotations_path


def main():
    # Download and extract dataset
    download_path = Path(os.path.expanduser("~/data")) / DATASET_SHORTNAME
    download_path.mkdir(parents=True, exist_ok=True)
    download_data(download_path)

    # Create COCO dataset from CSV annotations
    annotations_path = download_path / "combined_coco_annotations.json"
    images_path = download_path / "JPEGImages"
    images_path.mkdir(parents=True, exist_ok=True)
    create_coco_dataset(download_path, images_path, annotations_path)

    # Extract and save sample image
    image_example = build_and_visualize_supervision_dataset_from_coco_dataset(
        images_path, annotations_path
    )
    plt.imsave(f"{DATASET_SHORTNAME}_sample_image.png", image_example)


# if __name__ == "__main__":
#     main()
