# %%
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import json
import shutil
import os
from dotenv import load_dotenv

from data_preview.utils import visualize_supervision_dataset, download_file, extract_downloaded_file

load_dotenv()

DATASET_SHORTNAME = "roboflow_fish"
DATA_DIR = Path("/mnt/data/tmp/") / DATASET_SHORTNAME
IMAGES_DIR = DATA_DIR / "combined"
ANNOTATIONS_PATH = IMAGES_DIR / "annotations.json"


def join_all_images_and_annotations_into_single_coco_dataset(
    data_dir: Path, coco_images_dir: Path, coco_annotations_path: Path
):
    """
    Merges all train/val/test splits into a single COCO dataset.
    """
    splits = ["train", "valid", "test"]
    split_dirs = [data_dir / split for split in splits]

    # Check if all directories exist
    for split_dir in split_dirs:
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        annotation_file = split_dir / "_annotations.coco.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotation_file}")

    # Create output directory
    output_dir = coco_images_dir
    output_dir.mkdir(exist_ok=True)

    # Load and combine annotations
    combined_annotations = {"images": [], "annotations": [], "categories": []}
    image_id_offset = 0
    annotation_id_offset = 0

    for split_dir in split_dirs:
        annotation_file = split_dir / "_annotations.coco.json"
        with open(annotation_file) as f:
            split_annotations = json.load(f)

        # Update image IDs and copy images
        for image in split_annotations["images"]:
            old_image_id = image["id"]
            image["id"] += image_id_offset

            # Copy image file to combined directory
            src_path = split_dir / image["file_name"]
            dst_path = output_dir / image["file_name"]
            shutil.copy2(src_path, dst_path)

            combined_annotations["images"].append(image)

        # Update annotation IDs and image IDs
        for annotation in split_annotations["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
            combined_annotations["annotations"].append(annotation)

        # Only copy categories from first split since they should be the same
        if not combined_annotations["categories"]:
            combined_annotations["categories"] = split_annotations["categories"]

        image_id_offset = max(img["id"] for img in combined_annotations["images"]) + 1
        if combined_annotations["annotations"]:
            annotation_id_offset = (
                max(ann["id"] for ann in combined_annotations["annotations"]) + 1
            )

    # Save combined annotations
    with open(coco_annotations_path, "w") as f:
        json.dump(combined_annotations, f)

    print(f"Combined dataset saved to {output_dir}")
    print(f"Total images: {len(combined_annotations['images'])}")
    print(f"Total annotations: {len(combined_annotations['annotations'])}")

    return coco_images_dir, coco_annotations_path


def extract_example_image(images_path, annotations_path, dataset_shortname):
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_path),
        annotations_path=str(annotations_path),
    )

    image_example = visualize_supervision_dataset(dataset)

    if image_example is not None:
        output_path = Path(f"{dataset_shortname}_sample_image.png")
        plt.imsave(str(output_path), image_example)
        print(f"Sample image saved to {output_path}")
    else:
        print("No annotated images found to save as sample")


def download_data(data_dir):
    data_dir.mkdir(exist_ok=True, parents=True)

    roboflow_api_key = os.getenv("ROBOFLOW_KEY_ROBOFLOW_FISH")
    print(f"ROBOFLOW_KEY: {roboflow_api_key}")
    data_url = f"https://public.roboflow.com/ds/KJiCisn7wU?key={roboflow_api_key}"

    data_path = data_dir / "roboflow_fish.zip"

    if data_dir.exists() and len(list(data_dir.glob("*"))) > 0:
        print("Data already downloaded and extracted")
    else:
        print("Downloading data...")
        download_file(data_url, data_path)
        print("Extracting data...")
        extract_downloaded_file(data_path, data_dir)


def main():
    download_data(DATA_DIR)

    # Create combined dataset
    coco_images_dir = DATA_DIR / "combined"
    coco_annotations_path = coco_images_dir / "annotations.json"
    coco_images_dir, coco_annotations_path = join_all_images_and_annotations_into_single_coco_dataset(
        DATA_DIR, coco_images_dir, coco_annotations_path
    )

    extract_example_image(coco_images_dir, coco_annotations_path, DATASET_SHORTNAME)


# if __name__ == "__main__":
#     main()
