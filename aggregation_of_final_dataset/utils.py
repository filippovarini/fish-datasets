import json
import shutil
from pathlib import Path
from typing import Callable, Any

from aggregation_of_final_dataset.settings import Settings


settings = Settings()


def compress_annotations_to_single_category(
    annotations_path: Path, category_names_to_keep: list[str], output_path: Path
):
    """
    Discards all annotations except for the ones in the category_names_to_keep list.
    For the ones that are kept, it renames all categories to a single category, fish.
    """
    # Check if new annotation file already exists
    if output_path.exists():
        print(f"New annotation file already exists at {output_path}")
        return output_path

    # Load the annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Check existing categories
    all_category_names = {c["name"] for c in coco_data["categories"]}
    print(
        f"Found categories {all_category_names} but only keeping {category_names_to_keep}"
    )

    # Filter annotations to only include the ones in the category_names_to_keep list
    new_annotations = []
    for annotation in coco_data["annotations"]:
        annotation_category = coco_data["categories"][annotation["category_id"]]
        assert (
            annotation["category_id"] == annotation_category["id"]
        ), f"Annotation category_id is {annotation['category_id']} not {annotation_category['id']}"

        if annotation_category["name"] in category_names_to_keep:
            annotation["category_id"] = Settings.coco_category_id
            new_annotations.append(annotation)

    # Print the number of annotations before and after compression
    original_annotation_count = len(coco_data["annotations"])
    compressed_annotation_count = len(new_annotations)
    print(f"Original annotation count: {original_annotation_count}")
    print(f"Compressed annotation count: {compressed_annotation_count}")

    # Compress categories to a single category
    coco_data["categories"] = Settings.coco_categories
    coco_data["annotations"] = new_annotations

    # Store the new annotation file
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)


def split_coco_dataset_into_train_validation(
    source_images_path: Path,
    source_annotations_path: Path,
    train_dataset_path: Path,
    val_dataset_path: Path,
    get_split_for_image: Callable[[str], bool]
) -> None:
    """
    Splits a COCO dataset into training and validation sets based on a provided function.

    Args:
        source_images_path: Path to the source images
        source_annotations_path: Path to the source annotations
        train_dataset_path: Path to the training dataset
        val_dataset_path: Path to the validation dataset
        get_split_for_image: Function that accepts image filename and returns True for training set images
    """
    # Assert all paths are valid
    if not source_images_path.exists():
        raise FileNotFoundError(f"Images folder not found at {source_images_path}")
    if not source_annotations_path.exists():
        raise FileNotFoundError(
            f"COCO annotations file not found at {source_annotations_path}"
        )
    if not train_dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_dataset_path}")
    if not val_dataset_path.exists():
        raise FileNotFoundError(f"Validation dataset not found at {val_dataset_path}")

    # Create output directories
    train_images_path = train_dataset_path / settings.images_folder_name
    val_images_path = val_dataset_path / settings.images_folder_name

    train_images_path.mkdir(parents=True)
    val_images_path.mkdir(parents=True)

    # Load COCO annotations
    with open(source_annotations_path, "r") as f:
        coco_data = json.load(f)

    # Initialize new COCO datasets
    train_coco = {
        "images": [],
        "annotations": [],
        "categories": Settings.coco_categories,
    }

    val_coco = {"images": [], "annotations": [], "categories": Settings.coco_categories}

    # Keep track of image IDs in each split
    train_image_ids = set()
    val_image_ids = set()

    # Split images based on the provided function
    for i, image in enumerate(coco_data.get("images", [])):
        # Get the image filename
        filename = image.get("file_name")

        # Use the provided function to determine if this is a training image
        is_train = get_split_for_image(filename)

        # Determine target paths
        if is_train:
            target_coco = train_coco
            target_image_ids = train_image_ids
            target_images_path = train_images_path
        else:
            target_coco = val_coco
            target_image_ids = val_image_ids
            target_images_path = val_images_path

        # Add image to the appropriate split
        target_image_ids.add(image["id"])
        target_coco["images"].append(image)

        # Copy the image file to the target directory
        source_image_path = source_images_path / Path(filename).name
        target_image_path = target_images_path / Path(filename).name

        assert (
            source_image_path.exists()
        ), f"Source image not found: {source_image_path}"

        # only copy if the target image does not exist
        if not target_image_path.exists():
            print(f"Copying image {i} of {len(coco_data.get('images', []))}", end="\r")
            shutil.copy2(source_image_path, target_image_path)

    print()

    # Split annotations based on image IDs
    for annotation in coco_data.get("annotations", []):
        # Ensure all annotations have already been filtered to only include the fish category
        assert (
            annotation["category_id"] == Settings.coco_category_id
        ), f"Annotation category_id is {annotation['category_id']} not {Settings.coco_category_id}"

        image_id = annotation["image_id"]

        if image_id in train_image_ids:
            train_coco["annotations"].append(annotation)
        elif image_id in val_image_ids:
            val_coco["annotations"].append(annotation)

    # Save the COCO annotations to files
    with open(train_dataset_path / settings.coco_file_name, "w") as f:
        json.dump(train_coco, f, indent=2)

    with open(val_dataset_path / settings.coco_file_name, "w") as f:
        json.dump(val_coco, f, indent=2)

    # Print summary
    print(f"Split dataset into:")
    print(
        f"  - Training: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations"
    )
    print(
        f"  - Validation: {len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations"
    )
