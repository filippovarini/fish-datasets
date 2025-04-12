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

from data_preview.utils import download_and_extract


DATASET_SHORTNAME = "fishclef"
DATA_URL = "https://zenodo.org/records/15202605/files/fishclef_2015_release.zip?download=1"


def download_data(data_dir: Path):
    """Download and extract the fishclef dataset"""
    data_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract(data_dir, DATA_URL, DATASET_SHORTNAME)


def convert_xml_to_coco(xml_file, output_dir=None):
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
    video_id = root.get("id")

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


def convert_annotations(data_dir: Path):
    """Convert XML annotations to COCO format for both training and test sets"""
    training_annotations_path = data_dir / "fishclef_2015_release/training_set/gt"
    training_annotations_coco_path = data_dir / "fishclef_2015_release/training_set/gt_coco"
    training_annotations_coco_path.mkdir(exist_ok=True, parents=True)
    
    test_annotations_path = data_dir / "fishclef_2015_release/test_set/gt" 
    test_annotations_coco_path = data_dir / "fishclef_2015_release/test_set/gt_coco"
    test_annotations_coco_path.mkdir(exist_ok=True, parents=True)
    
    # Convert training annotations
    xml_files = glob.glob(os.path.join(training_annotations_path, "*.xml"))
    for xml_file in xml_files:
        output_json = convert_xml_to_coco(xml_file, training_annotations_coco_path)
        print(f"Converted '{xml_file}' to '{output_json}'.")

    # Convert test annotations
    xml_files = glob.glob(os.path.join(test_annotations_path, "*.xml"))
    for xml_file in xml_files:
        output_json = convert_xml_to_coco(xml_file, test_annotations_coco_path)
        print(f"Converted '{xml_file}' to '{output_json}'.")


def extract_frame(cap, frame_index):
    """Extract a specific frame from a video"""
    # Set video position to the desired frame index and read it
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Frame {frame_index} could not be read")
    return frame


def detections_from_coco(coco_data, image_id):
    """
    Converts COCO annotations for a given image_id into a supervision.Detections object.
    Assumes bounding boxes in COCO are in [x, y, w, h] format.
    """
    anns = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
    boxes = []
    confidences = []
    class_ids = []
    
    for ann in anns:
        x, y, w, h = ann["bbox"]
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        boxes.append([x, y, x + w, y + h])
        confidences.append(1.0)  # No score provided; assume full confidence.
        class_ids.append(ann["category_id"])
    
    if boxes:
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)
    else:
        boxes = np.empty((0, 4))
        confidences = np.empty((0,))
        class_ids = np.empty((0,))
    
    return sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)


def visualize_annotated_frames_grid(video_path, coco_json_path, output_path=None, num_images=16, grid_shape=(4,4)):
    """Visualize annotated frames from a video in a grid"""
    # Load COCO annotations
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)
    
    # Build a set of image_ids that have at least one annotation
    annotated_frame_ids = {ann["image_id"] for ann in coco_data["annotations"]}
    
    # Filter the images list to only those with annotations, sorted by image_id
    annotated_images = sorted([img for img in coco_data["images"] if img["id"] in annotated_frame_ids],
                              key=lambda x: x["id"])
    
    if len(annotated_images) == 0:
        print("No annotated frames found.")
        return
    
    # Choose num_images frames at equal intervals from annotated_images
    if len(annotated_images) < num_images:
        selected_images = annotated_images
    else:
        indices = np.linspace(0, len(annotated_images) - 1, num_images, dtype=int)
        selected_images = [annotated_images[i] for i in indices]
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    # Initialize BoxAnnotator from supervision
    box_annotator = sv.BoxAnnotator()
    
    annotated_frames = []
    selected_ids = []  # to keep track of the selected frame ids
    
    for image_info in selected_images:
        frame_id = image_info["id"]
        # Extract the frame using frame_id (assuming image_id matches frame index)
        frame = extract_frame(cap, frame_id)
        
        # Get detections for this frame
        detections = detections_from_coco(coco_data, image_id=frame_id)
        
        # Annotate the frame
        annotated = box_annotator.annotate(scene=frame, detections=detections)
        
        # Convert from BGR (OpenCV) to RGB for matplotlib display
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated_frames.append(annotated)
        selected_ids.append(frame_id)
    
    cap.release()
    
    # Plot the selected frames in a grid
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img in enumerate(annotated_frames):
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Frame {selected_ids[i]}")
    
    # Hide any remaining subplots if necessary
    for j in range(len(annotated_frames), len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    
    # Save the figure
    if output_path is None:
        output_path = f"{DATASET_SHORTNAME}_sample_image.png"
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def extract_example_image(data_dir: Path):
    """Extract and visualize example images from training and test sets"""
    training_videos_path = data_dir / "fishclef_2015_release/training_set/videos"
    training_annotations_coco_path = data_dir / "fishclef_2015_release/training_set/gt_coco"
    
    test_videos_path = data_dir / "fishclef_2015_release/test_set/videos"
    test_annotations_coco_path = data_dir / "fishclef_2015_release/test_set/gt_coco"
    
    # Visualise a video from training set
    video_files = glob.glob(os.path.join(training_videos_path, "*.flv"))
    if video_files:
        video_file = random.choice(video_files)
        coco_annotations = training_annotations_coco_path / (os.path.splitext(os.path.basename(video_file))[0] + ".json")
        if coco_annotations.exists():
            print(f"Visualizing training video: {video_file}")
            visualize_annotated_frames_grid(video_file, coco_annotations, f"{DATASET_SHORTNAME}_training_sample_image.png")
    
    # Visualise a video from test set
    video_files = glob.glob(os.path.join(test_videos_path, "*.flv"))
    if video_files:
        video_file = random.choice(video_files)
        coco_annotations = test_annotations_coco_path / (os.path.splitext(os.path.basename(video_file))[0] + ".json")
        if coco_annotations.exists():
            print(f"Visualizing test video: {video_file}")
            visualize_annotated_frames_grid(video_file, coco_annotations, f"{DATASET_SHORTNAME}_test_sample_image.png")


def main():
    data_dir = Path("/mnt/data/dev/fish-datasets/data/raw") / DATASET_SHORTNAME
    data_dir.mkdir(parents=True, exist_ok=True)
    
    download_data(data_dir)
    convert_annotations(data_dir)
    extract_example_image(data_dir)


if __name__ == "__main__":
    main()