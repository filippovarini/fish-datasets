from typing import List
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import random
import os
import glob
import xml.etree.ElementTree as ET
import json
import cv2
import numpy as np

from data_preview.utils import download_and_extract, build_and_visualize_supervision_dataset_from_coco_dataset


DATASET_SHORTNAME = "fishclef"
DATA_URL = "https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1"


def download_data(data_dir: Path):
    """Download and extract the fishclef dataset"""
    data_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract(data_dir, DATA_URL, DATASET_SHORTNAME)


def get_width_and_heigth_of_video(videos_dir: Path, video_id: str):
    """Get the width and height of a video"""
    video_path = videos_dir / f"{video_id}.flv"
    cap = cv2.VideoCapture(str(video_path))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return width, height


def convert_xml_to_coco(videos_dir: Path, xml_file, output_dir=None):
    """
    Converts a single XML annotation file to COCO JSON format.
    
    Parameters:
        xml_file (str): Path to the XML file.
        output_dir (str): Directory where the JSON file will be saved. 
                          If None, the JSON is saved in the same directory as xml_file.
    
    Returns:
        output_json (str): Path to the generated COCO JSON file.
    """
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the video id/name from the <video> element.
    video_id = Path(xml_file).stem
    
    width, height = get_width_and_heigth_of_video(videos_dir, video_id)

    images = []
    annotations = []
    categories = {}
    ann_id = 1

    # Process each <frame> element
    for frame in root.findall("frame"):
        frame_id = frame.get("id")
        # Create image entry; using the video id and frame id in the file name.
        image_info = {
            "id": int(frame_id),
            "file_name": f"{video_id}_frame_{frame_id}.jpg",
            "width": width,
            "height": height,
        }
        images.append(image_info)

        # Process each <object> in the frame
        for obj in frame.findall("object"):
            species = obj.get("fish_species")
            if species not in categories:
                categories[species] = len(categories) + 1  # assign new id

            # Extract bounding box coordinates: top-left x, top-left y, width, height
            x = int(obj.get("x"))
            y = int(obj.get("y"))
            w = int(obj.get("w"))
            h = int(obj.get("h"))

            ann = {
                "id": ann_id,
                "image_id": int(frame_id),
                "category_id": categories[species],
                "bbox": [x, y, w, h],
            }
            annotations.append(ann)
            ann_id += 1

    # Create categories list for COCO format
    categories_list = [
        {"id": cat_id, "name": species}
        for species, cat_id in categories.items()
    ]

    # Assemble the final COCO dictionary
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories_list
    }

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(xml_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    output_json = os.path.join(output_dir, base_name + ".json")
    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=4)

    return output_json


def convert_annotations(download_dir: Path, output_dir: Path):
    """Convert XML annotations to COCO format for both training and test sets"""
    training_annotations_path = download_dir / "fishclef_2015_release" / "training_set" / "gt"
    training_annotations_coco_path = output_dir / "fishclef_2015_release" / "training_set" / "gt_coco"
    training_videos_path = download_dir / "fishclef_2015_release" / "training_set" / "videos"

    test_annotations_path = download_dir / "fishclef_2015_release" / "test_set" / "gt"
    test_annotations_coco_path = output_dir / "fishclef_2015_release" / "test_set" / "gt_coco"
    test_videos_path = download_dir / "fishclef_2015_release" / "test_set" / "videos"

    if training_annotations_coco_path.exists() and test_annotations_coco_path.exists():
        print("Annotations already converted, skipping")
        return
    
    training_annotations_coco_path.mkdir(exist_ok=True, parents=True)
    test_annotations_coco_path.mkdir(exist_ok=True, parents=True)
    
    
    # Convert training annotations
    xml_files = training_annotations_path.rglob("*.xml")
    for xml_file in xml_files:
        output_json = convert_xml_to_coco(training_videos_path, xml_file, training_annotations_coco_path)
        print(f"Converted '{xml_file}' to '{output_json}'.")

    # Convert test annotations
    xml_files = test_annotations_path.rglob("*.xml")
    for xml_file in xml_files:
        output_json = convert_xml_to_coco(test_videos_path, xml_file, test_annotations_coco_path)
        print(f"Converted '{xml_file}' to '{output_json}'.")


def extract_frame(cap, frame_index):
    """Extract a specific frame from a video"""
    # Set video position to the desired frame index and read it
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Frame {frame_index} could not be read")
    return frame


def merge_coco_datasets_into_single_dataset(annotations_paths: List[Path], output_path: Path):
    """
    Merges a list of COCO datasets into a single COCO dataset.
    
    Args:
        annotations_paths: List of paths to COCO annotation files
        output_path: Path to save the merged COCO dataset
    """
    # Validate inputs
    if not annotations_paths:
        raise ValueError("No annotation paths provided")
    
    if output_path.exists():
        print(f"Output file already exists at {output_path}")
        return output_path
    
    # Initialize counters and data structures
    image_id_counter = 1  # 1-indexed
    annotation_id_counter = 1  # 1-indexed
    category_id_counter = 1  # 1-indexed
    category_names_to_id = {}
    
    merged_coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    
    # Process each dataset
    for path in annotations_paths:
        print(f"Merging {path}")
        if not path.exists():
            print(f"Warning: File {path} does not exist, skipping")
            continue
            
        with open(path, "r") as f:
            try:
                coco_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: File {path} is not valid JSON, skipping")
                continue
        
        old_image_id_to_new_image_id = {}
        old_category_id_to_new_category_id = {}
        
        # Process images
        for image in coco_data.get("images", []):
            old_image_id = image["id"]
            old_image_id_to_new_image_id[old_image_id] = image_id_counter
            image["id"] = image_id_counter
            merged_coco["images"].append(image)
            image_id_counter += 1
        
        # Process categories
        unique_categories = []
        for category in coco_data.get("categories", []):
            old_category_id = category["id"]
            category_name = category["name"]
            
            if category_name in category_names_to_id:
                # Category already exists, just map the ID
                old_category_id_to_new_category_id[old_category_id] = category_names_to_id[category_name]
            else:
                # New category
                category_names_to_id[category_name] = category_id_counter
                old_category_id_to_new_category_id[old_category_id] = category_id_counter
                
                # Update category ID and add to list of unique categories
                category["id"] = category_id_counter
                unique_categories.append(category)
                
                category_id_counter += 1
        
        # Add unique categories to merged dataset
        merged_coco["categories"].extend(unique_categories)
        
        # Process annotations
        for annotation in coco_data.get("annotations", []):
            try:
                old_image_id = annotation["image_id"]
                old_category_id = annotation["category_id"]
                
                # Map to new IDs
                annotation["image_id"] = old_image_id_to_new_image_id[old_image_id]
                annotation["category_id"] = old_category_id_to_new_category_id[old_category_id]
                annotation["id"] = annotation_id_counter
                
                merged_coco["annotations"].append(annotation)
                annotation_id_counter += 1
            except KeyError as e:
                print(f"Warning: Invalid annotation (missing {e}), skipping")
    
    # Print summary
    print(f"Merged dataset contains:")
    print(f"  - {len(merged_coco['images'])} images")
    print(f"  - {len(merged_coco['annotations'])} annotations")
    print(f"  - {len(merged_coco['categories'])} categories")
    
    # Save merged dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged_coco, f, indent=2)
    
    return output_path


def extract_frames_from_videos(download_dir: Path, frames_dir: Path, coco_data: dict):
    """
    Extract frames from videos and save them to disk
    
    Args:
        download_dir: Directory containing videos
        frames_dir: Directory to save extracted frames
        coco_data: COCO dataset containing frame information
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    if frames_dir.exists():
        print(f"Frames directory already exists at {frames_dir}")
        return frames_dir
    
    video_name_to_video_path = {video_path.stem: video_path for video_path in download_dir.rglob("*.flv")}
    print(f"Found {len(video_name_to_video_path)} videos")
    
    # Track which frames we have already extracted to avoid duplicates
    extracted_frames = set()
    
    # Extract frames from coco annotations
    for image_info in coco_data["images"]:
        frame_filename = image_info["file_name"]
        frame_id = int(Path(frame_filename).stem.split("_frame_")[1])
        frame_name = Path(frame_filename).stem.split("_frame_")[0]
        if frame_name not in video_name_to_video_path:
            print(f"⚠️ Frame {frame_name} not found in {video_name_to_video_path}")
            continue

        video_path = video_name_to_video_path[frame_name]
        print(f"Extracting frame {frame_id} from {video_path}", end="\r")
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Error opening video file: {video_path}")
            continue
        
        # Extract the frame
        try:
            frame = extract_frame(cap, frame_id)
            # Save the frame
            output_path = frames_dir / frame_filename
            cv2.imwrite(str(output_path), frame)
            extracted_frames.add(str(output_path))
        except Exception as e:
            print(f"⚠️ Error extracting frame {frame_id} from {video_path}: {e}")
        
        # Release video capture
        cap.release()
    
    print("\n")
    print(f"Extracted {len(extracted_frames)} frames")
    return frames_dir


def main():
    data_dir = Path("/mnt/data/dev/fish-datasets/data/raw") / DATASET_SHORTNAME
    data_dir.mkdir(parents=True, exist_ok=True)
    
    download_data(data_dir)
    convert_annotations(data_dir, data_dir)

    merged_coco_path = data_dir / "fishclef_2015_release/merged_annotations.json"
    frames_dir = data_dir / "fishclef_2015_release/extracted_frames"
    
    
    # Merge all annotations
    training_videos_path = data_dir / "fishclef_2015_release/training_set/videos"
    training_annotations_coco_path = data_dir / "fishclef_2015_release/training_set/gt_coco"
    
    test_videos_path = data_dir / "fishclef_2015_release/test_set/videos"
    test_annotations_coco_path = data_dir / "fishclef_2015_release/test_set/gt_coco"
    
    # Get all annotation files
    training_annotations = list(training_annotations_coco_path.glob("*.json"))
    test_annotations = list(test_annotations_coco_path.glob("*.json"))
    all_annotations = training_annotations + test_annotations
    
    if not all_annotations:
        print("No annotation files found")
        return
    
    # Merge all annotations
    print(f"Merging {len(all_annotations)} annotation files")
    merged_coco_path = merge_coco_datasets_into_single_dataset(all_annotations, merged_coco_path)
    
    extract_frames_from_videos(data_dir, frames_dir, merged_coco_path)
    
    build_and_visualize_supervision_dataset_from_coco_dataset(
        images_dir=frames_dir,
        annotations_path=merged_coco_path
    )


# if __name__ == "__main__":
#     main()