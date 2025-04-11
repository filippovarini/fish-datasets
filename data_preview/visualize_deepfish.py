"""
Process and visualise the DeepFish dataset

This dataset includes count, classification, and segmentation labels; we are only 
using the segmentation labels in this script, and because we are training a
detector, we are reducing them to boxes. Segmentation labels are stored as images,
not as text, so we need to parse the connected components from the images.
"""
import os
import random
import json
from pathlib import Path
import supervision as sv
import cv2
from skimage import measure
from tqdm import tqdm
from dotenv import load_dotenv

from data_preview.utils import visualize_supervision_dataset, download_and_extract_zip

load_dotenv()

DATASET_SHORTNAME = "deepfish"
DATA_DIR = Path(os.path.expanduser("~/data")) / DATASET_SHORTNAME
SEGMENTATION_BASE = DATA_DIR / "DeepFish" / "Segmentation"
SEGMENTATION_MASK_BASE = SEGMENTATION_BASE / "masks" / "valid"
SEGMENTATION_IMAGE_BASE = SEGMENTATION_BASE / "images" / "valid"
SOURCE_URL = "http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar"
COCO_DATASET_FILE = DATA_DIR / "deepfish_coco.json"


def get_boxes_from_mask_image(mask_file):
    """
    Load a binary image, find connected components, and convert to COCO-formatted bounding boxes.
    
    Args:
        mask_file (str): Path to the binary image file
        
    Returns:
        dict: COCO format annotations
    """
    # Read the image
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

    image_id = os.path.splitext(os.path.basename(mask_file))[0]
    
    # Ensure binary image (threshold if not already binary)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    labels = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labels)
    
    # Prepare COCO-formatted annotations
    annotations = []
    for idx, region in enumerate(regions):
        # Get bounding box (y1, x1, y2, x2)
        bbox = region.bbox
        
        # Convert to COCO format [x, y, width, height]
        coco_bbox = [
            bbox[1],                # x
            bbox[0],                # y
            bbox[3] - bbox[1],      # width
            bbox[2] - bbox[0]       # height
        ]
        
        # Create annotation entry
        annotation = {
            "id": f"{image_id}_{str(idx).zfill(3)}",
            "image_id": image_id,
            "category_id": 1,
            "bbox": coco_bbox,            
        }
        annotations.append(annotation)
    
    return annotations


def create_coco_dataset():
    """
    Process mask images to create a COCO format dataset
    """
    # Ensure directories exist
    if not SEGMENTATION_MASK_BASE.exists():
        raise FileNotFoundError(f"Mask directory not found: {SEGMENTATION_MASK_BASE}")
    
    if not SEGMENTATION_IMAGE_BASE.exists():
        raise FileNotFoundError(f"Image directory not found: {SEGMENTATION_IMAGE_BASE}")
    
    # Enumerate mask files
    valid_masks = list(SEGMENTATION_MASK_BASE.glob("*"))
    print(f"Found {len(valid_masks)} mask files")
    
    # Enumerate image files
    valid_images = list(SEGMENTATION_IMAGE_BASE.glob("*"))
    print(f"Found {len(valid_images)} image files")
    
    assert len(valid_images) == len(valid_masks), "Number of images and masks should match"
    
    # Convert mask images to bounding boxes
    annotation_records = []
    debug_max_file = None  # Set to a number to limit processing for debugging
    
    for i_mask, mask_file in tqdm(enumerate(valid_masks), total=len(valid_masks)):
        if debug_max_file is not None and i_mask > debug_max_file:
            break
            
        coco_formatted_annotations = get_boxes_from_mask_image(mask_file)
        annotation_records.extend(coco_formatted_annotations)
    
    print(f"Created {len(annotation_records)} annotations")
    
    # Create complete COCO dataset
    coco_data = {
        "info": {},
        "categories": [{"name": "fish", "id": 1}],
        "annotations": annotation_records,
        "images": []
    }
    
    for image_file_abs in tqdm(valid_images):
        im = {}
        im_cv = cv2.imread(str(image_file_abs))
        image_id = os.path.splitext(os.path.basename(image_file_abs))[0]
        im["id"] = image_id
        im["file_name"] = str(image_file_abs.relative_to(SEGMENTATION_IMAGE_BASE.parent))
        im["height"] = im_cv.shape[0]
        im["width"] = im_cv.shape[1]
        
        coco_data["images"].append(im)
    
    # Save COCO dataset
    with open(COCO_DATASET_FILE, "w") as f:
        json.dump(coco_data, f, indent=1)
        
    print(f"COCO dataset saved to {COCO_DATASET_FILE}")
    
    return SEGMENTATION_IMAGE_BASE, COCO_DATASET_FILE


def extract_example_image(images_path, annotations_path, dataset_shortname):
    """
    Create and save a sample visualized image
    """
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_path),
        annotations_path=str(annotations_path),
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset classes: {dataset.classes}")
    
    # Create a visualization with grid
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_images = []
    for _ in range(16):
        i = random.randint(0, len(dataset) - 1)
        _, image, annotations = dataset[i]
        labels = [dataset.classes[class_id] for class_id in annotations.class_id]
        
        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
        annotated_images.append(annotated_image)
    
    sv.plot_images_grid(
        annotated_images,
        grid_size=(4, 4),
        titles=None,
        size=(20, 12),
        cmap="gray"
    )
    
    # Save a single sample image
    i_image = min(100, len(dataset) - 1)
    _, image, annotations = dataset[i_image]
    labels = [dataset.classes[class_id] for class_id in annotations.class_id]
    
    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
    
    sample_image_output_file = DATA_DIR / f"{dataset_shortname}_sample.jpg"
    cv2.imwrite(str(sample_image_output_file), annotated_image)
    print(f"Sample image saved to {sample_image_output_file}")
    
    return annotated_image


def main():
    # Download and extract dataset
    download_and_extract_zip(DATA_DIR, SOURCE_URL, DATASET_SHORTNAME)
    
    # Create COCO dataset from mask images
    images_path, annotations_path = create_coco_dataset()
    
    # Extract and save sample image
    extract_example_image(images_path, annotations_path, DATASET_SHORTNAME)


if __name__ == "__main__":
    main()