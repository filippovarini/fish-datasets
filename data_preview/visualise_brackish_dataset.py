# %%
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import random
import json
import zipfile
import requests
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
# %%
dataset_shortname = "brackish_dataset"
data_dir = Path("/mnt/data/tmp/") / dataset_shortname
data_dir.mkdir(exist_ok=True, parents=True)

# %%
roboflow_api_key = os.getenv("ROBOFLOW_KEY_BRAKISH")
data_url = f"https://public.roboflow.com/ds/vGBLxigwno?key={roboflow_api_key}"
data_path = data_dir / "brakish_dataset.zip"


def download_file(url, save_path):
    """Download a file from URL to specified path"""
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def extract_zip(zip_path, extract_to):
    """Extract zip file to specified directory"""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


if data_dir.exists() and len(list(data_dir.glob("*"))) > 0:
    print("Data already downloaded and extracted")
else:
    print("Downloading data...")
    download_file(data_url, data_path)
    print("Extracting data...")
    extract_zip(data_path, data_dir)
    print("Removing Zipped files...")
    data_path.unlink()


# %%
def join_all_images_and_annotations_into_single_dataset(data_dir):
    # Define paths to the train, validation, and test directories
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
    output_dir = data_dir / "combined"
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
    output_annotation_file = output_dir / "annotations.json"
    with open(output_annotation_file, "w") as f:
        json.dump(combined_annotations, f)

    print(f"Combined dataset saved to {output_dir}")
    print(f"Total images: {len(combined_annotations['images'])}")
    print(f"Total annotations: {len(combined_annotations['annotations'])}")

    return output_dir, output_annotation_file


images_dir, annotations_file = join_all_images_and_annotations_into_single_dataset(
    data_dir
)

# %%
dataset = sv.DetectionDataset.from_coco(
    images_directory_path=str(images_dir),
    annotations_path=str(annotations_file),
)

print(f"Dataset length: {len(dataset)}")
print(f"Dataset classes: {dataset.classes}")


# %%
def compute_random_grid_of_annotated_images():
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    image_example = None
    annotated_images = []

    # Get a sample of images to visualize
    num_samples = min(16, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)

    for i in sample_indices:
        _, image, annotations = dataset[i]

        # Skip if no annotations
        if len(annotations) == 0:
            continue

        labels = [dataset.classes[class_id] for class_id in annotations.class_id]

        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
        annotated_images.append(annotated_image)

        if image_example is None and len(annotations) > 0:
            image_example = annotated_image

    # Fill remaining slots if needed
    while len(annotated_images) < 16:
        i = random.randint(0, len(dataset) - 1)
        _, image, annotations = dataset[i]

        labels = [dataset.classes[class_id] for class_id in annotations.class_id]

        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
        annotated_images.append(annotated_image)

        if image_example is None and len(annotations) > 0:
            image_example = annotated_image

    sv.plot_images_grid(annotated_images, grid_size=(4, 4), titles=None, size=(20, 12))

    return image_example


image_example = compute_random_grid_of_annotated_images()

# %%
# Save a sample image
if image_example is not None:
    output_path = Path(f"data_preview/{dataset_shortname}_sample_image.png")
    plt.imsave(str(output_path), image_example)
    print(f"Sample image saved to {output_path}")
else:
    print("No annotated images found to save as sample")

# %%
