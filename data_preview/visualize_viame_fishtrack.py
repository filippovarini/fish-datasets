# %%
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import cv2

from utils import download_and_extract_zip

# %%
dataset_shortname = "viame_fishtrack"
data_dir = Path("/mnt/data/tmp/") / dataset_shortname
data_dir.mkdir(exist_ok=True)

# %%
# Download the dataset
data_url = "https://viame.kitware.com/api/v1/dive_dataset/export?folderIds=[%2265a1a1d1cf5a99794eaacb57%22,%2265a1a291cf5a99794eab01fb%22,%2265a1a205cf5a99794eaadbb6%22,%2265a1a223cf5a99794eaae509%22,%2265a1a20ccf5a99794eaadddd%22,%2265a1a1d1cf5a99794eaacb3d%22,%2265a1a23ecf5a99794eaaed79%22,%2265a1a20ccf5a99794eaadde0%22,%2265a1a223cf5a99794eaae50e%22,%2265a1a1d1cf5a99794eaacb52%22,%2265a1a28fcf5a99794eab01b2%22,%2265a1a22fcf5a99794eaae8c1%22,%2265a1a205cf5a99794eaadbbb%22,%2265a1a1ffcf5a99794eaad9c8%22,%2265a1a1d8cf5a99794eaacd93%22,%2265a1a1f1cf5a99794eaad548%22,%2265a1a1d1cf5a99794eaacb67%22,%2265a1a23ecf5a99794eaaed82%22,%2265a1a230cf5a99794eaae92a%22,%2265a1a244cf5a99794eaaef6b%22]"
data_path = data_dir / f"{dataset_shortname}.zip"

data_dir = download_and_extract_zip(data_dir, data_url, dataset_shortname)


# %%
def build_image_id(video_path: Path, timestamp_str: str):
    return f"{video_path.stem}_{timestamp_str}"


def extract_frame_from_video(frames_path: Path, video_path: Path, timestamp_str: str):
    # Create the frames directory if it doesn't exist
    frames_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.strptime(timestamp_str, "%H:%M:%S.%f")
    image_id = build_image_id(video_path, timestamp_str)
    output_path = frames_path / f"{image_id}.jpg"

    print(f"Extracting frame from {video_path} at {timestamp}")

    # Extract the frames
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp.timestamp() * 1000)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Failed to read frame from {video_path}")
    height, width, _ = frame.shape
    cv2.imwrite(output_path, frame)
    cap.release()

    print(f"Saved frame to {output_path}")
    return output_path, height, width


def viame_annotations_to_coco(camera_path: Path, output_dir: Path):
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
        frame_id = build_image_id(video_path, frame_timestamp)

        # Add image entry if we haven't seen this frame before
        if frame_id not in image_ids:
            image_ids.add(frame_id)
            image_filename, image_height, image_width = extract_frame_from_video(
                output_frames_path, video_path, frame_timestamp
            )
            coco_data["images"].append(
                {
                    "id": frame_id,
                    "file_name": image_filename,
                    "height": image_height,
                    "width": image_width,
                }
            )

        # Extract category information (species)
        species = row["10-11+: Repeated Species"]
        assert not pd.isna(species), f"Species is NaN for row {index}"

        # Add new category if not seen before
        if species not in categories:
            categories[species] = category_id
            coco_data["categories"].append(
                {
                    "id": category_id,
                    "name": species,
                }
            )
            category_id += 1
        
        # Extract bounding box coordinates
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


output_dir = Path("/mnt/data/dev/fish-datasets/tmp/test_CDFW-LakeCam-April-Tules3")
coco_data = viame_annotations_to_coco(
    data_dir / "CDFW-LakeCam-April-Tules3", output_dir
)

with open(output_dir / "annotations.json", "w") as f:
    json.dump(coco_data, f)


# %%
