# This script downloads, merges and visualize a sample of two datasets: Marine Detect FishInv and Marine Detect Megafauna
# These two datasets were merged as there were many common images with annotations divided between the two datasets. 


import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import random

import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from IPython.display import HTML
from PIL import Image
import numpy as np
import base64
import shutil
import json
import cv2
import os
import io
import subprocess

from aggregation_of_final_dataset.settings import Settings
from data_preview.utils import build_and_visualize_supervision_dataset_from_coco_dataset, save_sample_image

settings = Settings()

DATASET_SHORTNAME = "marine_detect"

DATASETS = ["marine_detect_megafauna", "marine_detect_fishinv"]


data_URLs = {
    "marine_detect_megafauna": "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/MegaFauna-dataset.zip", 
    "marine_detect_fishinv": "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/FishInv-dataset.zip"
    }



download_path = settings.raw_dir / DATASET_SHORTNAME

# Next two variables refer to the final dataset consisting in the merge of both megafauna and fishinv
final_images_path = download_path / "images"
final_annotations_path = download_path / "annotations.json"

categories_fishinv = [
    {"id": 1, "name": "bolbometopon_muricatum"},
    {"id": 2, "name": "chaetodontidae"},
    {"id": 3, "name": "cheilinus_undulatus"},
    {"id": 4, "name": "cromileptes_altivelis"},
    {"id": 5, "name": "fish"},
    {"id": 6, "name": "haemulidae"},
    {"id": 7, "name": "lutjanidae"},
    {"id": 8, "name": "muraenidae"},
    {"id": 9, "name": "scaridae"},
    {"id": 10, "name": "serranidae"},
    {"id": 11, "name": "urchin"},
    {"id": 12, "name": "giant_clam"},
    {"id": 13, "name": "sea_cucumber"},
    {"id": 14, "name": "crown_of_thorns"},
    {"id": 15, "name": "lobster"}
    ]

categories_megafauna = [
        {"id": 1, "name": "ray"},
        {"id": 2, "name": "shark"},
        {"id": 3, "name": "turtle"}
    ]



def download_data():
    # Download both datasets
    if (download_path.exists() and any(download_path.iterdir())):
        print(f"Download folder ({download_path}) already exists. Data will not be downloaded. Remove the folder to download data")
        return
    download_path.mkdir(exist_ok=True, parents=True)
    
    for dataset in DATASETS:

        single_dataset_folder = download_path / dataset
        single_dataset_folder.mkdir(exist_ok=True, parents=True)

        data_path = single_dataset_folder / "dataset.zip"
        data_url = data_URLs[dataset]

        subprocess.run(["wget", "-nc", "-O", str(data_path), data_url], check=True)
        subprocess.run(["unzip", "-q", str(data_path), "-d", single_dataset_folder], check=True)


def merge_files(root_dir, images_path, annotations_path):
    """

    Merges images and labels from train, valid, and test directories into
    single 'images/' and 'labels/' directories,
    adding dataset-specific suffixes. Skip OzFish images and empty annotation files.

    IMPORTANT: this function operates on a single dataset. The merge of the two datasets will be performed later 
    """

    # Define dataset splits
    datasets = [("train", "train"), ("valid", "valid"), ("test", "test")]

    # Create target directories if they don’t exist
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(annotations_path, exist_ok=True)

    ignored_count = 0  # Counter for ignored files
    empty_annotation_count = 0  # Counter for empty annotation files
    skip_patterns = ["_L.MP4.", "_R.MP4.", "_L.avi.", "_R.avi."]  # Patterns to skip
    # Files matching this pattern are skipped as they stem from the OzFish dataset and shouldn't be considered twice

    for dataset, suffix in datasets:
        img_src = os.path.join(root_dir, dataset, "images")
        lbl_src = os.path.join(root_dir, dataset, "labels")
        

        # Copy images
        if os.path.exists(img_src):
            for file in os.listdir(img_src):
                if any(pattern in file for pattern in skip_patterns):
                    ignored_count += 1
                    continue  # Skip this file

                if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Ensure it's an image
                    name, ext = os.path.splitext(file)
                    new_filename = f"{name}_{suffix}{ext}"
                    shutil.copy2(os.path.join(img_src, file), os.path.join(images_path, new_filename))

        # Copy labels
        if os.path.exists(lbl_src):
            for file in os.listdir(lbl_src):
                if any(pattern in file for pattern in skip_patterns):
                    ignored_count += 1
                    continue  # Skip this file

                if file.lower().endswith('.txt'):  # Ensure it's a label file
                    label_path = os.path.join(lbl_src, file)
                    name, ext = os.path.splitext(file)
                    new_filename = f"{name}_{suffix}{ext}"

                    # Check if the annotation file is empty
                    with open(label_path, 'r') as label_file:
                        if not label_file.read().strip():  # If file is empty
                            empty_annotation_count += 1
                            continue  # Skip this empty annotation file

                    shutil.copy2(label_path, os.path.join(annotations_path, new_filename))
                    

    print(f"All labels saved to {annotations_path}")
    print(f"Ignored {ignored_count} files stemming from the OzFish dataset.")
    print(f"Excluded {empty_annotation_count} empty annotation files.")



def convert_to_coco(image_dir, label_dir, output_json, categories):
    # ### Clean the annotations
    # Turn into COCO format readable by `supervision` library, for easy visualization and conversion to other formats.
    # - annotations.json only contains annotations for images with at least one bounding box
    """
    Converts annotation text files into a COCO format JSON.

    Parameters:
    - image_dir: Path to the directory containing images.
    - label_dir: Path to the directory containing annotation text files.
    - output_json: Path to save the output JSON file.
    """

    # Initialize COCO JSON structure
    coco_data = {
        "info": {
            "description": "Dataset in COCO format",
            "version": "1.0",
            "year": 2025
        },
        "images": [],
        "annotations": [],
        "categories": categories
    }

    # Counters for unique IDs
    image_id = 0
    annotation_id = 0

    # Process each image file
    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Ensure it's an image
            continue

        image_name = os.path.basename(image_file)
        image_path = os.path.join(image_dir, image_name)
        annotation_file = os.path.splitext(image_name)[0] + ".txt"
        annotation_path = os.path.join(label_dir, annotation_file)

        # Get image size
        with Image.open(image_path) as img:
            width, height = img.size

        # Add image metadata to COCO format
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Check if annotation file exists and is not empty
        if os.path.exists(annotation_path) and os.path.getsize(annotation_path) > 0:
            with open(annotation_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Skipping invalid line in {annotation_file}: {line}")
                        continue

                    class_id, x_center_rel, y_center_rel, bbox_width_rel, bbox_height_rel = map(float, parts)
                    # Denormalizing
                    x_center_abs = x_center_rel * width
                    y_center_abs = y_center_rel * height
                    bbox_width_abs = bbox_width_rel * width
                    bbox_height_abs = bbox_height_rel * height

                    # Calcola x_min, y_min
                    x_min_abs = x_center_abs - bbox_width_abs / 2
                    y_min_abs = y_center_abs - bbox_height_abs / 2

                    bbox = [x_min_abs, y_min_abs, bbox_width_abs, bbox_height_abs]


                    # Add annotation entry
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": (int(class_id)+1),
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],  # width * height
                        "iscrowd": 0
                    })
                    annotation_id += 1

        # Move to the next image
        image_id += 1

    # Save to JSON file
    with open(output_json, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(f"COCO annotations saved to {output_json}")




def create_coco_datasets():
    # Create 2 different coco datasets
    for dataset in DATASETS:
        images_path = download_path / dataset / "images"
        annotations_path = download_path / dataset / "labels"
        annotations_file = download_path / dataset /"annotations.json"

        if dataset == "marine_detect_fishinv":
            merge_files(str(download_path / dataset / "notebooks/datasets/FishInvSplit"), images_path, annotations_path)
            convert_to_coco(images_path, annotations_path, annotations_file, categories_fishinv)
        else:
            merge_files(str(download_path / dataset / "notebooks/datasets/MegaFaunaSplit"), images_path, annotations_path)
            convert_to_coco(images_path, annotations_path, annotations_file, categories_megafauna)

    
        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        cleaned_annotations = []

        print(f"Number of annotations: {len(annotations['annotations'])}")

        for i, annotation in enumerate(annotations["annotations"]):
            if "bbox" not in annotation or len(annotation["bbox"]) == 0:
                print(f"No bbox found for {annotation['image_id']}")
            else:
                cleaned_annotations.append(annotation)

        annotations["annotations"] = cleaned_annotations

        with open(annotations_file, "w") as f:
            print(f"Number of annotations: {len(annotations['annotations'])}")
            json.dump(annotations, f)


def get_max_id(json, field):
    id_list = []
    for element in json[field]:
        id_list.append(element["id"])
    return max(id_list)


def merge_datasets():
    # This function merges the two datasets (megafuna and fishinv)
    # First dataset will be copied in the final folder and then the second dataset will be added by modifying the json and adding its images

    # Copying json and images of megafauna into final folder
    shutil.copy(download_path / DATASETS[0] / "annotations.json", final_annotations_path)
    shutil.copytree(download_path / DATASETS[0] / "images", final_images_path)

    
    addedd_annotations = 0
    number_images_already_present = 0

    with open(final_annotations_path, 'r', encoding='utf-8') as annotations_file:
        final_json = json.load(annotations_file)

    with open(download_path / DATASETS[1] / "annotations.json", 'r', encoding='utf-8') as annotations_file:
        second_dataset_annotations = json.load(annotations_file)


    print("Initial annotations: ", len(final_json["annotations"]))
    print("Second dataset annotations: ", len(second_dataset_annotations["annotations"]))


    max_image_id = get_max_id(final_json, "images")
    max_annotation_id = get_max_id(final_json, "annotations")

    print("Max image ID before merging is: ", max_image_id)
    print("Max annotation ID before merging is: ", max_annotation_id)

    category_offset = len(final_json["categories"]) # new categories will have id = old_id + offset (works because in this case the datasets have got 0 common categories)
    

    for image in second_dataset_annotations["images"]:
        old_image_id = image["id"]
        image_already_present = False

        for image_of_final_dataset in final_json["images"]:

            if image["file_name"].rsplit("_", 1)[0] == image_of_final_dataset["file_name"].rsplit("_", 1)[0]:
                # Image already exists. Need to add the new annotations
                number_images_already_present += 1
                image_already_present = True
                new_image_id = image_of_final_dataset["id"]
                break # No need to control other images  

        if image_already_present == False:
            shutil.copy(
                download_path / DATASETS[1] / "images" / image["file_name"], 
                final_images_path / image["file_name"]
            )
            
            new_image_id = max_image_id + 1

            max_image_id = max_image_id + 1
            
            # adding the image in the final json
            new_image = image
            new_image["id"] = new_image_id
            final_json["images"].append(new_image)


        second_dataset_annotations_copy = second_dataset_annotations["annotations"][:]
        # Copy annotations
        for annotation in second_dataset_annotations_copy:
            if annotation["image_id"] == old_image_id:
                
                new_annotation = annotation
                new_annotation["image_id"] = new_image_id
                new_annotation["id"] = max_annotation_id
                new_annotation["category_id"] = annotation["category_id"] + category_offset

                final_json["annotations"].append(new_annotation)
                second_dataset_annotations["annotations"].remove(annotation)

                max_annotation_id = max_annotation_id + 1
                addedd_annotations = addedd_annotations + 1
    print("Number of addedd annotations:" + str(addedd_annotations))

    # Modify categories
    new_categories = second_dataset_annotations["categories"]
    for category in new_categories:
        category["id"] = category["id"] + category_offset
    final_json["categories"].extend(new_categories)

    with open(final_annotations_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    print(f"Number of final annotations: {len(final_json['annotations'])}")
    print(f"Number of final images: {len(final_json['images'])}")
    print(f"Number of final categories: {len(final_json['categories'])}")
    print(f"Number of images already present: {number_images_already_present}")



if __name__ == "__main__":
    
    download_data()
    create_coco_datasets()
    merge_datasets()
    
    # Visualize data and save sample image
    image_sample = build_and_visualize_supervision_dataset_from_coco_dataset(
        final_images_path, 
        final_annotations_path)
    save_sample_image(DATASET_SHORTNAME, image_sample)