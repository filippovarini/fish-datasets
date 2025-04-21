# data_preview/visualize_vmat.py
from pathlib import Path
import os
import random
import json
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import shutil
import supervision as sv

from data_preview.utils import (
    visualize_supervision_dataset,
    download_file,
    extract_downloaded_file,
)

DATASET_SHORTNAME = "vmat"
# Alternative URL from Zenodo or other service. 
# Replace with an actual URL after the dataset is uploaded to Zenodo
DATA_URL = "https://warp.whoi.edu/vmat/"  # Replace with actual Zenodo URL


def download_vmat_dataset(output_path: Path):
    """
    Download the VMAT dataset from Zenodo
    """
    output_path.mkdir(exist_ok=True, parents=True)

    # Check if data already exists
    if output_path.exists() and len(list(output_path.glob("*"))) > 0:
        print(f"Data already exists at {output_path}")
        return output_path

    print(f"Downloading VMAT dataset to {output_path}")

    # This part would need to be updated with the actual download URL and approach
    # For example, if you've uploaded to Zenodo:
    download_url = DATA_URL
    download_path = output_path / f"{DATASET_SHORTNAME}.zip"

    try:
        download_file(download_url, download_path)
        extract_downloaded_file(download_path, output_path)
        return output_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download the dataset manually from https://warp.whoi.edu/vmat/")
        print(f"and extract it to {output_path}")
        return None


def convert_vmat_to_coco(
    data_dir: Path, output_annotations_path: Path, output_images_dir: Path
):
    """
    Convert VMAT dataset annotations to COCO format.
    """
    if output_annotations_path.exists():
        print(f"COCO annotations already exist at {output_annotations_path}")
        return output_images_dir, output_annotations_path

    # Path to images and annotations
    image_folder = (
        data_dir / "GX010090_shark.MP4_clip"
    )  # Update based on actual structure
    annotation_file = (
        data_dir / "GX010090_shark.MP4_clip_obj0.txt"
    )  # Update based on actual structure

    if not image_folder.exists() or not annotation_file.exists():
        print(
            f"Required files not found. Please ensure dataset was downloaded correctly."
        )
        return None, None

    # Create output directories
    output_images_dir.mkdir(exist_ok=True, parents=True)

    # Load annotations
    annotations = pd.read_csv(annotation_file, sep="\s+", header=None)
    annotations.columns = ["left", "top", "width", "height"]

    # Add frame number (assuming frames start at 1)
    annotations["frame"] = annotations.index + 1

    # Helper function to format frame numbers
    def frame_number_to_filename(frame_number):
        return f"{frame_number:05d}.jpg"  # This will convert 1 to 00001.jpg

    # Initialize COCO format
    coco_format = {
        "info": {"description": "VMAT Dataset"},
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "shark", "supercategory": "animal"}],
    }

    # Process frames and annotations
    annotation_id = 1

    for frame_idx in annotations.index:
        frame_annotation = annotations.iloc[frame_idx]
        frame_number = frame_annotation["frame"]
        image_filename = frame_number_to_filename(frame_number)
        image_path = image_folder / image_filename

        # Copy image to output directory
        if image_path.exists():
            shutil.copy(image_path, output_images_dir / image_filename)

            # Get image dimensions
            img = cv2.imread(str(image_path))
            if img is not None:
                height, width = img.shape[:2]

                # Add image info to COCO format
                coco_format["images"].append(
                    {
                        "id": int(frame_number),
                        "file_name": image_filename,
                        "width": width,
                        "height": height,
                    }
                )

                # Add annotation
                x, y = frame_annotation["left"], frame_annotation["top"]
                w, h = frame_annotation["width"], frame_annotation["height"]

                coco_format["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": int(frame_number),
                        "category_id": 1,  # Shark
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    # Save COCO format annotations
    with open(output_annotations_path, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"Converted VMAT dataset to COCO format")
    print(f"- Annotations saved to {output_annotations_path}")
    print(f"- Images saved to {output_images_dir}")
    print(f"- Total images: {len(coco_format['images'])}")
    print(f"- Total annotations: {len(coco_format['annotations'])}")

    return output_images_dir, output_annotations_path


def extract_example_image(images_path, annotations_path):
    """Extract and save a sample annotated image from the dataset."""
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_path),
        annotations_path=str(annotations_path),
    )

    image_example = visualize_supervision_dataset(dataset)

    if image_example is not None:
        output_path = Path(f"data_preview/{DATASET_SHORTNAME}_sample_image.png")
        plt.imsave(str(output_path), image_example)
        print(f"Sample image saved to {output_path}")
    else:
        print("No annotated images found to save as sample")


def main():
    """Main function to process and visualize the dataset."""
    # Set up data paths
    data_dir = Path("~/data").expanduser() / DATASET_SHORTNAME
    coco_dir = data_dir / "coco"
    coco_images_dir = coco_dir / "images"
    coco_annotations_path = coco_dir / "annotations.json"

    # Create directories
    coco_dir.mkdir(exist_ok=True, parents=True)
    coco_images_dir.mkdir(exist_ok=True, parents=True)

    # Download dataset
    download_path = download_vmat_dataset(data_dir)

    if download_path:
        # Convert to COCO format
        images_dir, annotations_path = convert_vmat_to_coco(
            download_path, coco_annotations_path, coco_images_dir
        )

        # Extract example image
        if images_dir and annotations_path:
            extract_example_image(images_dir, annotations_path)


if __name__ == "__main__":
    main()
