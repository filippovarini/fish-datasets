import requests
import zipfile
import random
from pathlib import Path

import supervision as sv


def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download complete: {save_path}")


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: {zip_path}")
    print("Removing Zipped files...")
    zip_path.unlink()
    
    
def download_and_extract_zip(data_dir, data_url, dataset_shortname):
    data_path = data_dir / f"{dataset_shortname}.zip"
    if data_dir.exists() and len(list(data_dir.glob("*"))) > 0:
        print("Data already downloaded and extracted")
    else:
        print("Downloading data...")
        download_file(data_url, data_path)
        print("Extracting data...")
        extract_zip(data_path, data_dir)
        print("Removing Zipped files...")
        data_path.unlink()
    return data_dir


def visualize_supervision_dataset(dataset, num_samples=16, grid_size=(4, 4), size=(20, 12)):
    """Visualize random samples from a dataset with bounding boxes and labels."""
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset classes: {dataset.classes}")
    
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
        annotated_images, grid_size=grid_size, titles=image_names, size=size, cmap="gray"
    )

    return image_example