# data_preview/visualize_viame_fishtrack.py
import json
import random
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import supervision as sv
import pandas as pd
import cv2

from utils import download_and_extract_zip


class VideoFrameExtractor:
    @staticmethod
    def build_image_id(video_path: Path, timestamp_str: str) -> str:
        """Generate a unique identifier for a video frame."""
        return f"{video_path.stem}_{timestamp_str}"

    @classmethod
    def extract_frame(cls, frames_path: Path, video_path: Path, timestamp_str: str):
        """Extract a frame from a video at the specified timestamp."""
        frames_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.strptime(timestamp_str, "%H:%M:%S.%f")
        image_id = cls.build_image_id(video_path, timestamp_str)
        output_path = frames_path / f"{image_id}.jpg"

        print(f"Extracting frame from {video_path} at {timestamp}")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp.timestamp() * 1000)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame from {video_path}")
        
        height, width, _ = frame.shape
        cv2.imwrite(str(output_path), frame)
        cap.release()

        print(f"Saved frame to {output_path}")
        return output_path, height, width


def viame_to_coco(camera_path: Path, output_dir: Path):
    """Convert VIAME annotations to COCO format."""
    csv_path = camera_path / "annotations.viame.csv"
    video_path = camera_path / f"{camera_path.name}.mp4"

    assert csv_path.exists(), f"CSV file not found: {csv_path}"
    assert video_path.exists(), f"Video file not found: {video_path}"

    # Load the CSV file - skip first row as it contains metadata
    df = pd.read_csv(csv_path, skiprows=lambda x: x in [1])

    output_frames_path = output_dir / "JPEGImages"
    output_frames_path.mkdir(parents=True, exist_ok=True)

    # Initialize COCO format dictionary
    coco_data = {"images": [], "annotations": [], "categories": []}
    
    # Keep track of categories and assigned IDs
    categories = {}
    category_id = 1

    # Track image IDs (frame numbers) we've already processed
    image_ids = set()
    
    # Track annotation ID
    annotation_id = 1

    for index, row in df.iterrows():
        frame_timestamp = row["2: Video or Image Identifier"]
        frame_id = VideoFrameExtractor.build_image_id(video_path, frame_timestamp)

        # Add image entry if we haven't seen this frame before
        if frame_id not in image_ids:
            image_ids.add(frame_id)
            image_filename, image_height, image_width = VideoFrameExtractor.extract_frame(
                output_frames_path, video_path, frame_timestamp
            )
            coco_data["images"].append({
                "id": frame_id,
                "file_name": str(image_filename),
                "height": image_height,
                "width": image_width,
            })

        # Process species information
        species = row["10-11+: Repeated Species"]
        assert not pd.isna(species), f"Species is NaN for row {index}"

        # Add new category if not seen before
        if species not in categories:
            categories[species] = category_id
            coco_data["categories"].append({
                "id": category_id,
                "name": species,
            })
            category_id += 1
        
        # Process bounding box coordinates
        bbox_cols = ["4-7: Img-bbox(TL_x", "TL_y", "BR_x", "BR_y)"]
        assert all(col in row.index for col in bbox_cols), f"Bounding box columns not found for row {index}"
        assert not any(pd.isna(row[col]) for col in bbox_cols), f"Bounding box values are NaN for row {index}"
        
        xmin = float(row["4-7: Img-bbox(TL_x"])
        ymin = float(row["TL_y"])
        xmax = float(row["BR_x"])
        ymax = float(row["BR_y)"])
        
        # COCO format uses [x,y,width,height] for bbox
        width = xmax - xmin
        height = ymax - ymin
        
        # Add annotation
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": frame_id,
            "category_id": categories[species],
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
        })
        
        annotation_id += 1
        
    return coco_data


def visualize_dataset(dataset, num_samples=16, grid_size=(4, 4), size=(20, 12)):
    """Visualize random samples from a dataset with bounding boxes and labels."""
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    image_example = None
    annotated_images = []
    
    for _ in range(num_samples):
        i = random.randint(0, len(dataset) - 1)  # Avoid index out of range
        
        _, image, annotations = dataset[i]
        labels = [dataset.classes[class_id] for class_id in annotations.class_id]
        
        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
        annotated_images.append(annotated_image)
        
        if len(annotations) > 0 and image_example is None:
            image_example = annotated_image
    
    sv.plot_images_grid(
        annotated_images, 
        grid_size=grid_size, 
        titles=None, 
        size=size, 
        cmap="gray"
    )
    
    return image_example


def main():
    # Dataset configuration
    dataset_shortname = "viame_fishtrack"
    data_dir = Path("/mnt/data/tmp/") / dataset_shortname
    data_dir.mkdir(exist_ok=True)
    
    # Download the dataset
    data_url = "https://viame.kitware.com/api/v1/dive_dataset/export?folderIds=[%2265a1a1d1cf5a99794eaacb57%22,%2265a1a291cf5a99794eab01fb%22,%2265a1a205cf5a99794eaadbb6%22,%2265a1a223cf5a99794eaae509%22,%2265a1a20ccf5a99794eaadddd%22,%2265a1a1d1cf5a99794eaacb3d%22,%2265a1a23ecf5a99794eaaed79%22,%2265a1a20ccf5a99794eaadde0%22,%2265a1a223cf5a99794eaae50e%22,%2265a1a1d1cf5a99794eaacb52%22,%2265a1a28fcf5a99794eab01b2%22,%2265a1a22fcf5a99794eaae8c1%22,%2265a1a205cf5a99794eaadbbb%22,%2265a1a1ffcf5a99794eaad9c8%22,%2265a1a1d8cf5a99794eaacd93%22,%2265a1a1f1cf5a99794eaad548%22,%2265a1a1d1cf5a99794eaacb67%22,%2265a1a23ecf5a99794eaaed82%22,%2265a1a230cf5a99794eaae92a%22,%2265a1a244cf5a99794eaaef6b%22]"
    data_dir = download_and_extract_zip(data_dir, data_url, dataset_shortname)
    
    # Output configuration
    output_dir = Path("/mnt/data/dev/fish-datasets/tmp/test_CDFW-LakeCam-April-Tules3")
    
    # Convert VIAME annotations to COCO format
    camera_path = data_dir / "CDFW-LakeCam-April-Tules3"
    coco_data = viame_to_coco(camera_path, output_dir)
    
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(coco_data, f)
    
    # Load and visualize the dataset
    annotations_path = output_dir / "annotations.json"
    images_path = output_dir / "JPEGImages"
    
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_path),
        annotations_path=str(annotations_path),
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset classes: {dataset.classes}")
    
    # Visualize random samples from the dataset
    image_example = visualize_dataset(dataset)
    
    # Save a sample image
    if image_example is not None:
        plt.imsave(f"{dataset_shortname}_sample_image.png", image_example)


if __name__ == "__main__":
    main()