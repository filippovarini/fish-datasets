# %%
import json
import random
from pathlib import Path
import shutil
from typing import Tuple
import matplotlib.pyplot as plt
import supervision as sv
import pandas as pd
import cv2

from data_preview.utils import download_and_extract


DATASET_SHORTNAME = "viame_fishtrack"
TESTING = False
all_species = set()


# Avoid these categories
def _is_non_fish(species: str) -> bool:
    return species.startswith("non_fish")


def build_image_id(video_path: Path, frame_id: str) -> str:
    """Generate a unique identifier for a video frame."""
    return f"{video_path.stem}_{frame_id}"


def timestamp_to_milliseconds(timestamp_str: str) -> int:
    """
    Convert a timestamp string in format HH:MM:SS.ffffff to milliseconds.

    Args:
        timestamp_str: String in format "HH:MM:SS.ffffff"

    Returns:
        Total milliseconds

    Example:
        "00:01:49.900000" -> 109900 (milliseconds)
    """
    # Parse the timestamp string
    hours, minutes, seconds = timestamp_str.split(":")
    seconds, microseconds = seconds.split(".")

    # Convert to integers
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    # If microseconds part has fewer than 6 digits, pad with zeros
    microseconds = int(microseconds.ljust(6, "0"))

    # Calculate total milliseconds
    total_milliseconds = (
        hours * 3600 * 1000  # hours to ms
        + minutes * 60 * 1000  # minutes to ms
        + seconds * 1000  # seconds to ms
        + microseconds // 1000  # microseconds to ms (integer division)
    )

    return total_milliseconds


def extract_frame(
    frames_path: Path, video_path: Path, timestamp_str: str, frame_id: str
):
    """Extract a frame from a video at the specified timestamp."""
    filename = f"{frame_id}.jpg"
    output_path = frames_path / filename
    frames_path.mkdir(parents=True, exist_ok=True)

    milliseconds = timestamp_to_milliseconds(timestamp_str)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, milliseconds)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(
            f"Failed to read frame {frame_id} from {video_path} at {timestamp_str}"
        )

    height, width, _ = frame.shape
    if not output_path.exists():
        cv2.imwrite(str(output_path), frame)

    cap.release()

    return filename, height, width  # Return only the filename


def get_frame_from_video(
    row: pd.Series,
    video_path: Path,
    output_frames_path: Path,
    coco_data: dict,
    image_ids: set,
):
    """
    In case where the data is from videos, this function extracts the frame
    and stores it to the output directory.
    """
    frame_timestamp = row["2: Video or Image Identifier"]
    frame_id = build_image_id(video_path, frame_timestamp)

    # Add image entry if we haven't seen this frame before
    if frame_id not in image_ids:
        image_ids.add(frame_id)
        image_filename, image_height, image_width = extract_frame(
            output_frames_path, video_path, frame_timestamp, frame_id
        )
        coco_data["images"].append(
            {
                "id": frame_id,
                "file_name": str(image_filename),
                "height": image_height,
                "width": image_width,
            }
        )

    return frame_id


def get_frame_from_images(
    row: pd.Series,
    camera_path: Path,
    output_frames_path: Path,
    coco_data: dict,
    image_ids: set,
):
    """
    In case where the data is from images, this function extracts the frame
    and stores it to the output directory.
    """
    annotatoin_frame_id = row["2: Video or Image Identifier"]
    frame_id = build_image_id(camera_path, annotatoin_frame_id)

    if frame_id not in image_ids:
        frame_path = camera_path / annotatoin_frame_id
        assert frame_path.exists(), f"Frame not found: {frame_path}"

        height, width, _ = cv2.imread(str(frame_path)).shape

        # Save frame
        image_ids.add(frame_id)
        new_frame_path = output_frames_path / frame_id
        shutil.copy(frame_path, new_frame_path)

        # Save frame to coco annotations
        coco_data["images"].append(
            {
                "id": frame_id,
                "file_name": str(new_frame_path),
                "height": height,
                "width": width,
            }
        )

    return frame_id


def viame_to_coco(camera_path: Path, images_dir: Path, coco_data: dict):
    """
    Converts VIAME annotations to COCO format.

    Args:
        camera_path: Path to the camera directory.
        images_dir: Path to the directory to save the images.
        coco_annotations: Dictionary to save the COCO annotations. We use the
        same dictionary for all cameras, to continuosly populate it.
    """
    global all_species

    csv_path = camera_path / "annotations.viame.csv"
    assert csv_path.exists(), f"CSV file not found: {csv_path}"

    # Check if data is in images or videos
    is_video = True
    video_path = camera_path / f"{camera_path.name}.mp4"
    if not video_path.exists():
        print(
            f"🎥 Video file not found: {video_path}, trying to use png images instead"
        )
        images_available = len(list(camera_path.glob("*.png")))
        assert images_available > 0, f"No png images found in {camera_path}"
        is_video = False
        print(f"Using {images_available} png images instead of video")

    # Load the CSV file - skip first row as it contains metadata
    df = pd.read_csv(csv_path, skiprows=lambda x: x in [1])

    output_frames_path = images_dir
    output_frames_path.mkdir(parents=True, exist_ok=True)

    # Track image IDs (frame numbers) we've already processed
    image_ids = set()

    # Track annotation ID
    annotation_id = 1

    for index, row in df.iterrows():
        if TESTING and index > 100:
            # for testing purposes don't analyze full video
            break

        # Skip non-fish categories
        species = row["10-11+: Repeated Species"]
        if species not in all_species:
            all_species.add(species)
            print(f"Processing species: {species}")
        if not pd.notna(species) or _is_non_fish(species):
            print(f"Skipping row because of non-fish category: {species}")
            continue

        try:
            if is_video:
                frame_id = get_frame_from_video(
                    row, video_path, output_frames_path, coco_data, image_ids
                )
            else:
                frame_id = get_frame_from_images(
                    row, camera_path, output_frames_path, coco_data, image_ids
                )
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

        # Process bounding box coordinates
        bbox_cols = ["4-7: Img-bbox(TL_x", "TL_y", "BR_x", "BR_y)"]
        assert all(
            col in row.index for col in bbox_cols
        ), f"Bounding box columns not found for row {index}"
        assert not any(
            pd.isna(row[col]) for col in bbox_cols
        ), f"Bounding box values are NaN for row {index}"

        xmin = float(row["4-7: Img-bbox(TL_x"])
        ymin = float(row["TL_y"])
        xmax = float(row["BR_x"])
        ymax = float(row["BR_y)"])

        # COCO format uses [x,y,width,height] for bbox
        width = xmax - xmin
        height = ymax - ymin

        # Add annotation
        coco_data["annotations"].append(
            {
                "id": annotation_id,
                "image_id": frame_id,
                "category_id": 1,  # We only keep one fish category
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )

        annotation_id += 1


def visualize_dataset(dataset, num_samples=16, grid_size=(4, 4), size=(20, 12)):
    """Visualize random samples from a dataset with bounding boxes and labels."""
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


def download_data_and_build_coco_dataset(
    raw_data_download_path: Path, coco_dataset_path: Path, data_url: str
) -> Tuple[Path, Path]:
    """
    Downloads both the VIAME Train and Test datasets and builds a single
    COCO dataset with all the data.
    """
    downloaded_data_path = download_and_extract(
        raw_data_download_path, data_url, DATASET_SHORTNAME
    )

    # Convert VIAME annotations to COCO format for all cameras
    fish_category = {"id": 1, "name": "fish"}  # We only keep one fish category
    coco_data = {"images": [], "annotations": [], "categories": [fish_category]}

    # Create output directories
    images_output_path = coco_dataset_path / "JPEGImages"
    images_output_path.mkdir()

    for camera_path in downloaded_data_path.glob("*"):
        print(f"📸 Processing camera: {camera_path}...")
        if not camera_path.is_dir():
            continue

        viame_to_coco(camera_path, images_output_path, coco_data)

        print(f"Images loaded in coco dataset: {len(coco_data['images'])}")

    # Save the COCO annotations
    annotations_path = coco_dataset_path / "annotations_coco.json"
    with open(annotations_path, "w") as f:
        json.dump(coco_data, f)

    return images_output_path, annotations_path


def main():
    # Dataset configuration
    raw_data_download_path = Path("/mnt/data/tmp/mfd") / DATASET_SHORTNAME
    raw_data_download_path.mkdir(exist_ok=True)

    # Here we only download the test data, for demonstration. To see how to download both datasets,
    # check the aggregation_of_final_dataset Directory
    test_data_url = "https://viame.kitware.com/api/v1/dive_dataset/export?folderIds=[%2265a1a1d1cf5a99794eaacb57%22,%2265a1a291cf5a99794eab01fb%22,%2265a1a205cf5a99794eaadbb6%22,%2265a1a223cf5a99794eaae509%22,%2265a1a20ccf5a99794eaadddd%22,%2265a1a1d1cf5a99794eaacb3d%22,%2265a1a23ecf5a99794eaaed79%22,%2265a1a20ccf5a99794eaadde0%22,%2265a1a223cf5a99794eaae50e%22,%2265a1a1d1cf5a99794eaacb52%22,%2265a1a28fcf5a99794eab01b2%22,%2265a1a22fcf5a99794eaae8c1%22,%2265a1a205cf5a99794eaadbbb%22,%2265a1a1ffcf5a99794eaad9c8%22,%2265a1a1d8cf5a99794eaacd93%22,%2265a1a1f1cf5a99794eaad548%22,%2265a1a1d1cf5a99794eaacb67%22,%2265a1a23ecf5a99794eaaed82%22,%2265a1a230cf5a99794eaae92a%22,%2265a1a244cf5a99794eaaef6b%22]"
    test_coco_dataset_path = (
        Path("/mnt/data/dev/fish-datasets/final_dataset") / DATASET_SHORTNAME / "test"
    )
    test_coco_dataset_path.mkdir(parents=True)

    images_output_path, annotations_path = download_data_and_build_coco_dataset(
        raw_data_download_path, test_coco_dataset_path, test_data_url
    )

    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_output_path),
        annotations_path=str(annotations_path),
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset classes: {dataset.classes}")

    # Visualize random samples from the dataset
    image_example = visualize_dataset(dataset)
    output_dir = Path("/mnt/data/dev/fish-datasets/data_preview")

    # Save a sample image
    if image_example is not None:
        plt.imsave(output_dir / f"{DATASET_SHORTNAME}_sample_image.png", image_example)


# if __name__ == "__main__":
#     main()
# # %%
