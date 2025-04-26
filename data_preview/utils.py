import requests
import zipfile
import tarfile
import random
from pathlib import Path
from enum import Enum
import matplotlib, cv2

import supervision as sv


class CompressionType(Enum):
    ZIP = "zip"
    TAR = "tar"


def download_file(url: str, save_path: Path):
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download complete: {save_path}")


def extract_downloaded_file(
    download_path: Path,
    extract_to: Path,
    compression_type: CompressionType = CompressionType.ZIP,
):
    print(f"Extracting {download_path} to {extract_to}")
    
    if not download_path.exists():
        print("Compressed file not found")
        return

    match compression_type:
        case CompressionType.ZIP:
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extraction complete: {download_path}")
        case CompressionType.TAR:
            with tarfile.open(download_path, "r") as tar_ref:
                tar_ref.extractall(extract_to)
            print(f"Extraction complete: {download_path}")
        case _:
            raise ValueError(f"Unsupported compression type: {compression_type}")

    print("Removing Zipped files...")
    download_path.unlink()


def download_and_extract(
    data_dir: Path,
    data_url: str,
    dataset_shortname: str,
    compression_type: CompressionType = CompressionType.ZIP,
):
    """
    Download and extract a dataset from a URL.
    """
    download_path = data_dir / f"{dataset_shortname}.{compression_type.value}"
    
    if data_dir.exists() and len(list(data_dir.glob("*"))) > 0:
        print("Data already downloaded and extracted")
    else:
        print("Downloading data...")
        download_file(data_url, download_path)
        print("Extracting data...")
        extract_downloaded_file(download_path, data_dir, compression_type)
    return data_dir


def get_annotation_count_from_supervision_dataset(dataset):
    # Method 1: Count all annotations across all images
    total_annotations = 0
    for image_name, annotations_list in dataset.annotations.items():
        total_annotations += len(annotations_list)

    return total_annotations


def visualize_supervision_dataset(
    dataset, num_samples=16, grid_size=(4, 4), size=(20, 12)
):
    """Visualize random samples from a dataset with bounding boxes and labels."""

    matplotlib.use("Qt5Agg")
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset classes: {dataset.classes}")
    print(
        f"Dataset annotation count: {get_annotation_count_from_supervision_dataset(dataset)}"
    )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    image_example = None
    annotated_images = []
    image_names = []

    for _ in range(num_samples):
        i = random.randint(0, len(dataset) - 1)  # Avoid index out of range

        image_path, image, annotations = dataset[i]
        labels = [dataset.classes[class_id] for class_id in annotations.class_id]

        # Get image name
        image_name = Path(image_path).stem
        image_names.append(image_name)

        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
        annotated_images.append(annotated_image)

        if len(annotations) > 0 and image_example is None:
            image_example = annotated_image

    sv.plot_images_grid(
        annotated_images,
        grid_size=grid_size,
        titles=image_names,
        size=size,
        cmap="gray",
    )

    return image_example


def build_and_visualize_supervision_dataset_from_coco_dataset(
    images_dir: Path, annotations_path: Path
):
    """
    Given the path to COCO annotations and images, build a Supervision dataset and visualize it.
    """
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=images_dir,
        annotations_path=annotations_path,
    )

    image_example = visualize_supervision_dataset(dataset)
    return image_example

def save_sample_image(dataset_shortname, image):

    data_preview_path =  Path(f"./data_preview/{dataset_shortname}_sample_image.png")
    cv2.imwrite(data_preview_path, image)
