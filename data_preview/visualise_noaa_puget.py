# %%
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import random
import json
import zipfile
import os
import requests

# %%
dataset_shortname = "noaa_puget"
data_dir = Path("/mnt/data/tmp/") / dataset_shortname
data_dir.mkdir(exist_ok=True, parents=True)

# %%
data_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-images.zip"
annotations_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-annotations-2023.08.19.zip"

data_path = data_dir / "images.zip"
annotations_path = data_dir / "annotations.zip"


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


print("Downloading data and annotations...")
download_file(data_url, data_path)
download_file(annotations_url, annotations_path)
print("Extracting data and annotations...")
extract_zip(data_path, data_dir)
extract_zip(annotations_path, data_dir)
print("Removing Zipped files...")
os.remove(data_path)
os.remove(annotations_path)

# %%
annotations_path = data_dir / "noaa_estuary_fish-2023.08.19.json"
images_path = data_dir / "JPEGImages"

# %%
with open(annotations_path, "r") as f:
    annotations = json.load(f)

cleaned_annotations = []

print(f"Number of annotations: {len(annotations['annotations'])}")

for i, annotation in enumerate(annotations["annotations"]):
    if "bbox" not in annotation or len(annotation["bbox"]) == 0:
        print(f"No bbox found for {annotation['image_id']}")
    else:
        cleaned_annotations.append(annotation)

annotations["annotations"] = cleaned_annotations

with open(annotations_path, "w") as f:
    print(f"Number of annotations: {len(annotations['annotations'])}")
    json.dump(annotations, f)


# %%
dataset = sv.DetectionDataset.from_coco(
    images_directory_path=str(images_path),
    annotations_path=str(annotations_path),
)

print(f"Dataset length: {len(dataset)}")
print(f"Dataset classes: {dataset.classes}")

# %%
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

image_example = None

annotated_images = []
for _ in range(16):
    i = random.randint(0, len(dataset))

    _, image, annotations = dataset[i]

    labels = [dataset.classes[class_id] for class_id in annotations.class_id]

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
    annotated_images.append(annotated_image)

    if len(annotations) > 0:
        image_example = annotated_image

sv.plot_images_grid(
    annotated_images, grid_size=(4, 4), titles=None, size=(20, 12), cmap="gray"
)

plt.imsave(f"{dataset_shortname}_sample_image.png", image_example)

# %%
