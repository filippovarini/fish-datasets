"""
Process and visualise the DeepFish dataset

This dataset includes count, classification, and segmentation labels; we are only
using the segmentation labels in this script, and because we are training a
detector, we are reducing them to boxes. Segmentation labels are stored as images,
not as text, so we need to parse the connected components from the images.
"""

import os
import random
import json
from pathlib import Path
import supervision as sv
import cv2
from skimage import measure
from tqdm import tqdm
from dotenv import load_dotenv

import matplotlib.pyplot as plt

from data_preview.utils import (
    build_and_visualize_supervision_dataset_from_coco_dataset,
    download_and_extract,
    CompressionType,
)

load_dotenv()

DATASET_SHORTNAME = "deepfish"
SOURCE_URL = "http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar"


def download_data(download_path: Path):
    """
    Download and extract the DeepFish dataset
    """
    download_and_extract(
        download_path, SOURCE_URL, DATASET_SHORTNAME, CompressionType.TAR
    )


def get_boxes_from_mask_image(mask_file):
    """
    Load a binary image, find connected components, and convert to COCO-formatted bounding boxes.

    Args:
        mask_file (str): Path to the binary image file

    Returns:
        dict: COCO format annotations
    """
    # Read the image
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

    image_id = os.path.splitext(os.path.basename(mask_file))[0]

    # Ensure binary image (threshold if not already binary)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find connected components
    labels = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labels)

    # Prepare COCO-formatted annotations
    annotations = []
    for idx, region in enumerate(regions):
        # Get bounding box (y1, x1, y2, x2)
        bbox = region.bbox

        # Convert to COCO format [x, y, width, height]
        coco_bbox = [
            bbox[1],  # x
            bbox[0],  # y
            bbox[3] - bbox[1],  # width
            bbox[2] - bbox[0],  # height
        ]

        # Create annotation entry
        annotation = {
            "id": f"{image_id}_{str(idx).zfill(3)}",
            "image_id": image_id,
            "category_id": 1,
            "bbox": coco_bbox,
        }
        annotations.append(annotation)

    return annotations


def create_coco_dataset(download_path: Path):
    """
    Process mask images to create a COCO format dataset
    """
    segmentation_base = download_path / "DeepFish" / "Segmentation"
    segmentation_mask_base = segmentation_base / "masks" / "valid"
    segmentation_image_base = segmentation_base / "images" / "valid"
    coco_dataset_file = download_path / "deepfish_coco.json"
    
    if coco_dataset_file.exists():
        print(f"COCO dataset already exists: {coco_dataset_file}")
        return segmentation_image_base, coco_dataset_file

    # Ensure directories exist
    if not segmentation_mask_base.exists():
        raise FileNotFoundError(f"Mask directory not found: {segmentation_mask_base}")

    if not segmentation_image_base.exists():
        raise FileNotFoundError(f"Image directory not found: {segmentation_image_base}")

    # Enumerate mask files
    valid_masks = list(segmentation_mask_base.glob("*"))
    print(f"Found {len(valid_masks)} mask files")

    # Enumerate image files
    valid_images = list(segmentation_image_base.glob("*"))
    print(f"Found {len(valid_images)} image files")

    assert len(valid_images) == len(
        valid_masks
    ), "Number of images and masks should match"

    # Convert mask images to bounding boxes
    annotation_records = []
    debug_max_file = None  # Set to a number to limit processing for debugging

    for i_mask, mask_file in tqdm(enumerate(valid_masks), total=len(valid_masks)):
        if debug_max_file is not None and i_mask > debug_max_file:
            break

        coco_formatted_annotations = get_boxes_from_mask_image(mask_file)
        annotation_records.extend(coco_formatted_annotations)

    print(f"Created {len(annotation_records)} annotations")

    # Create complete COCO dataset
    coco_data = {
        "info": {},
        "categories": [{"name": "fish", "id": 0}],
        "annotations": annotation_records,
        "images": [],
    }

    for image_file_abs in tqdm(valid_images):
        im = {}
        im_cv = cv2.imread(str(image_file_abs))
        image_id = os.path.splitext(os.path.basename(image_file_abs))[0]
        im["id"] = image_id
        im["file_name"] = str(
            image_file_abs.relative_to(segmentation_image_base.parent)
        )
        im["height"] = im_cv.shape[0]
        im["width"] = im_cv.shape[1]

        coco_data["images"].append(im)

    # Save COCO dataset
    with open(coco_dataset_file, "w") as f:
        json.dump(coco_data, f, indent=1)

    print(f"COCO dataset saved to {coco_dataset_file}")

    return segmentation_image_base, coco_dataset_file


def main():
    # Download and extract dataset
    download_path = Path(os.path.expanduser("~/data")) / DATASET_SHORTNAME
    download_data(download_path)

    # Create COCO dataset from mask images
    images_path, annotations_path = create_coco_dataset(download_path)

    # Extract and save sample image
    random_image = build_and_visualize_supervision_dataset_from_coco_dataset(
        images_path, annotations_path
    )
    plt.imsave(f"{DATASET_SHORTNAME}_sample_image.png", random_image)


if __name__ == "__main__":
    main()
