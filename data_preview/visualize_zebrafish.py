from pathlib import Path
from typing import Tuple
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from PIL import Image, ImageDraw
import pandas as pd

from data_preview.utils import download_and_extract, CompressionType

DATASET_SHORTNAME = "zebrafish"
DATASET_URL = "aalborguniversity/aau-zebrafish-reid"


def download_data(download_path: Path):
    """Download the dataset directly from Kaggle."""
    if download_path.exists() and any(download_path.iterdir()):
        print("Data already downloaded in the directory:", download_path)
        return download_path

    print("Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download(DATASET_URL)
    shutil.move(path, str(download_path))
    return download_path


def clean_annotations_and_get_df(data_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """Clean and process the annotations."""
    data_path = data_dir / "2" / "data"
    annotation_path = data_dir / "2" / "annotations.csv"

    # Read the CSV file
    data_df = pd.read_csv(annotation_path, sep=";")

    # Check if the length of the dataframe is equal to the number of images in the data directory
    if len(list(data_path.iterdir())) == len(data_df):
        print(
            "Number of images in the data directory and the dataframe are equal:",
            len(data_df),
        )
    else:
        print("Number of images in the data directory and the dataframe are not equal")
        print("Number of images in the data directory:", len(list(data_path.iterdir())))
        print("Number of annotations in the dataframe:", len(data_df))

    # Process the combined columns
    combined_col = "Right,Turning,Occlusion,Glitch"
    for idx, col in enumerate(combined_col.split(",")):
        data_df[col] = data_df[combined_col].apply(lambda x: x.split(",")[idx])

    # Calculate bbox dimensions
    ws = data_df["Lower right corner X"] - data_df["Upper left corner X"]
    hs = data_df["Lower right corner Y"] - data_df["Upper left corner Y"]

    # Create bbox column
    data_df["bbox"] = [
        [x, y, w, h]
        for x, y, w, h in list(
            zip(
                data_df["Upper left corner X"].values,
                data_df["Upper left corner Y"].values,
                ws,
                hs,
            )
        )
    ]

    # Process other columns for COCO format
    data_df["path"] = data_path / data_df["Filename"]
    data_df["Object ID"] = data_df["Object ID"].astype(str)
    data_df["label"] = data_df["Annotation tag"]
    data_df["image_id"] = data_df["Filename"].apply(lambda x: x.split(".")[0])

    # Select relevant columns and group by image_id
    data_df = data_df[["image_id", "label", "bbox", "path"]]
    data_df = (
        data_df.groupby("image_id")
        .agg({"label": list, "bbox": list, "path": list})
        .reset_index()
    )

    return data_df, data_path


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj


def dataframe_to_coco(df, output_json_path: Path):
    """Convert the DataFrame to COCO format and save to JSON."""
    if output_json_path.exists():
        print(f"COCO format JSON already exists at {output_json_path}")
        return output_json_path
    
    # Initialize COCO format dictionary
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Create a mapping from label to category_id
    unique_labels = sorted(set([label for sublist in df["label"] for label in sublist]))
    label_to_id = {label: i + 1 for i, label in enumerate(unique_labels)}

    # Populate categories
    for label, cat_id in label_to_id.items():
        coco_format["categories"].append(
            {"id": cat_id, "name": label, "supercategory": "none"}
        )

    # Initialize annotation id
    annotation_id = 1

    # Iterate over each row in the dataframe
    for idx, row in df.iterrows():
        image_id = row["image_id"]
        image_path = row["path"][0]

        # Open image to get width and height
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image info to COCO format
        coco_format["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        # Add annotations for each object in the image
        for label, bbox in zip(row["label"], row["bbox"]):
            coco_format["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_to_id[label],
                    "bbox": convert_to_serializable(
                        bbox
                    ),  #  bbox is [x_min, y_min, width, height]
                    "area": bbox[2] * bbox[3],  # width * height
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    # Save the COCO format dictionary to a JSON file
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO format JSON saved to {output_json_path}")
    return output_json_path


def visualize_image_with_boxes(image_path, annotations, coco_data):
    """Visualize an image with its bounding boxes."""
    # Load the image
    image = Image.open(image_path)

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes and labels
    for ann in annotations:
        bbox = ann["bbox"]  # COCO bbox format: [x_min, y_min, width, height]
        category_id = ann["category_id"]
        category_name = next(
            (
                cat["name"]
                for cat in coco_data["categories"]
                if cat["id"] == category_id
            ),
            "unknown",
        )

        # Draw the bounding box
        x_min, y_min, width, height = map(int, bbox)
        draw.rectangle(
            [x_min, y_min, x_min + width, y_min + height], outline="red", width=2
        )

        # Draw the label
        label = f"{category_name}"
        draw.text((x_min, y_min - 15), label, fill="red")

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def get_image_annotations(image_id, coco_data):
    """Get all annotations for a specific image ID."""
    annotations = []
    for ann in coco_data["annotations"]:
        if ann["image_id"] == image_id:
            annotations.append(ann)
    return annotations


def visualize_random_images(coco_data, data_path, num_images=5):
    """Visualize random images with their bounding boxes."""
    import random

    images = coco_data["images"]
    random.shuffle(images)  # Shuffle to pick random images

    for i in range(min(num_images, len(images))):
        image_info = images[i]
        image_id = image_info["id"]
        image_path = (
            data_path / image_info["file_name"]
        )  # Assuming file_name contains the full path

        # Get annotations for the image
        annotations = get_image_annotations(image_id, coco_data)

        # Visualize the image with bounding boxes
        print(f"Visualizing image: {image_path}")
        visualize_image_with_boxes(image_path, annotations, coco_data)


def main():
    # Define paths
    download_path = Path("./dataset") / DATASET_SHORTNAME
    download_path.mkdir(exist_ok=True, parents=True)

    # Download data
    download_data(download_path)

    # Check if data directory exists
    if not download_path.exists():
        print(f"Data directory {download_path} does not exist")
        return

    # Clean and process annotations
    data_df, data_path = clean_annotations_and_get_df(download_path)

    # Convert to COCO format
    output_json_path = download_path / "coco_format.json"
    coco_path = dataframe_to_coco(data_df, output_json_path)

    # Load COCO annotations
    with open(coco_path, "r") as f:
        coco_data = json.load(f)

    # Visualize random images with bounding boxes
    visualize_random_images(coco_data, data_path, num_images=5)
