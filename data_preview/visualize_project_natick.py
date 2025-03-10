# %%
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import os
import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import random

# %%
dataset_shortname = "project_natick"
data_dir = Path("/mnt/data/tmp/") / dataset_shortname
data_dir.mkdir(exist_ok=True, parents=True)

# %%
data_url = "https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip"
data_path = data_dir / "data_release.zip"


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


print("Downloading data...")
download_file(data_url, data_path)
print("Extracting data...")
extract_zip(data_path, data_dir)
print("Removing Zipped files...")
os.remove(data_path)

# %%
# Set paths to the dataset
dataset_path = data_dir / "data_release" / "fish_annotations"
images_path = dataset_path / "JPEGImages"
annotations_path = dataset_path / "Annotations"


# %%
def add_extension_to_filename(directory, extension=".jpg"):
    # The XML files have a filename element that does not have an extension.
    # This function adds the extension to the filename.
    print(f"Checking and updating XML annotations in {directory}...")
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            filepath = Path(directory) / filename
            tree = ET.parse(filepath)
            root = tree.getroot()

            filename_element = root.find("filename")
            if filename_element is not None:
                if not filename_element.text.endswith(extension):
                    filename_element.text += extension
                    tree.write(filepath)
                    print(f"Updated {filename}")
                else:
                    print(f"{filename} already has extension, skipped.")


add_extension_to_filename(annotations_path)

# %%
# Load dataset using Supervision's VOC dataset loader
print("Loading dataset...")
dataset = sv.DetectionDataset.from_pascal_voc(
    images_directory_path=str(images_path),
    annotations_directory_path=str(annotations_path),
)

print(f"Dataset length: {len(dataset)}")
print(f"Dataset classes: {dataset.classes}")

# %%
# Visualize random samples
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_images = []
image_example = None

num_samples = min(16, len(dataset))
sample_indices = random.sample(range(len(dataset)), num_samples)

for i in sample_indices:
    _, image, detections = dataset[i]

    # Create labels
    labels = [dataset.classes[class_id] for class_id in detections.class_id]

    # Annotate image
    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)
    annotated_images.append(annotated_image)

    if len(detections) > 0 and image_example is None:
        image_example = annotated_image

# Plot grid of images
sv.plot_images_grid(annotated_images, grid_size=(4, 4), titles=None, size=(20, 12))

# %%
# Save a sample image
if image_example is not None:
    plt.imsave(f"{dataset_shortname}_sample_image.png", image_example)
    print(f"Sample image saved as {dataset_shortname}_sample_image.png")
else:
    print("No annotated images with detections found to save as example")

# %%
