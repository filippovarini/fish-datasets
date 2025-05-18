how_to_download = f"""

This script assumes that you have downloaded the zip file.

Download link: https://studentiunict-my.sharepoint.com/personal/simone_palazzo_unict_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsimone%5Fpalazzo%5Funict%5Fit%2FDocuments%2FDatasets%5Freleased%2Ff4k%5Fdetection%5Ftracking%2Ezip&parent=%2Fpersonal%2Fsimone%5Fpalazzo%5Funict%5Fit%2FDocuments%2FDatasets%5Freleased&ga=1

Once downloaded, put it in fish-dataset/data/raw/f4k

NOTE: This script requires ffmpeg

"""
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import cv2
import supervision as sv
import zipfile
import matplotlib.pyplot as plt
import tqdm

from data_preview.utils import (
    build_and_visualize_supervision_dataset_from_coco_dataset,
)
from aggregation_of_final_dataset.settings import Settings


settings = Settings()

DATASET_SHORTNAME = "f4k"

data_folder = settings.raw_dir / DATASET_SHORTNAME
zip_file = data_folder / "f4k_detection_tracking.zip"
extract_dir = label_dir = data_folder / "f4k_detection_tracking" / "f4k_detection_tracking"
output_dir = data_folder / "coco"
mp4_dir = data_folder / "mp4"
annotations_path = output_dir / "annotations_coco.json"


# Next dictionary defines the following information for each video:
#   - frame per second
#   - frame offset
#   - list of frames to be discarded
#
# These information are fundamental as frame extraction is based on framerate, but the dataset videos often have actual
# framerates different from the nominal value.
#
# The actual framerate have been found with: ffprobe -select_streams v -show_frames -of csv -f lavfi "movie=fish-datasets/data/raw/f4k/f4k_detection_tracking/gt_114.flv"
# The output of this command shows the nominal value, but counting the frame timestamps (right below the fps information) it is possible to find real value.
#
# Offset and discard_list have been found by analyzing sample output images after conversion to coco
video_info = {
    "106": [5, 0, []],
    "107": [5, 0, [1]],
    "109": [4, 0, []],
    "110": [5, 0, [1]],
    "111": [5, 0, [1]],
    "112": [5, 0, [1]],
    "113": [5, 0, [1]],
    "114": [5, -1, [1]],
    "116": [5, -1, [1]],
    "117": [5, 0, []],
    "118": [5, -1, [1]],
    "119": [5, 0, [0]],
    "120": [5, -1, [0]],
    "121": [5, -1, [1]],
    "122": [14.05, 0, [1]],
    "123": [24, 0, [0]],
    "124": [8, 0, [1]],
}


def check_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise NotImplementedError("Please install ffmpeg")


def extract_data():
    check_ffmpeg()

    # Extracts data from zipfile, create required folders and convert videos to mp4
    if not zip_file.exists():
        print("Missing file ", zip_file)
        raise NotImplementedError(how_to_download)

    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(mp4_dir, exist_ok=True)

    print("Extracting data...")

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Optional.
    os.remove(zip_file)

    # For ease of processing, we convert `.flv` into `.mp4`.
    print("Converting videos to mp4...")

    for input_path in tqdm.tqdm(extract_dir.rglob("*.flv")):
        if not input_path.name.startswith("."):
            print(f"Converting {input_path.name} to mp4...")
            output_path = mp4_dir / input_path.with_suffix(".mp4").name

            if output_path.exists():
                print(f"Skipping {input_path.name} because it already exists")
                continue

            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                output_path,
            ]

            subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )  # delete last two parameters to view ffmpeg conversion output


def get_all_categories():
    return {
        category: id
        for id, category in enumerate(
            (
                "fish",
                "open_sea",
                "sea",
                "rocks",
                "coral",
                "plant",
                "dark_area",
                "other",
                "algae",
            )
        )
    }


def extract_keyframes(
    label_path, video_path, video_name, category_id_map, annotation_id, image_id
):
    # Extract frames and their annotations from a video
    # Load data from xml
    tree = ET.parse(label_path)
    root = tree.getroot()

    # Create video data in coco format
    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    video_id = video_name.split("_")[1]

    for frame in root.findall("frame"):

        frame_id = (
            int(frame.attrib["id"]) + video_info[video_id][1]
        )  # Frame id = frame reference in the xml + offset

        if frame_id in video_info[video_id][2]:
            # Discard frame if in the discard list
            continue

        file_name = f"video_{video_name}_frame_{frame_id}.jpg"
        save_path = os.path.join(output_dir, file_name)

        # Calculate timestamp of the frame in the video (i.e. when the frame is displayed)
        timestamp = frame_id / video_info[video_id][0]

        cmd = [
            "ffmpeg",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            save_path,
        ]
        subprocess.run(
            cmd, stderr=subprocess.DEVNULL
        )  # delete last parameter to view ffmpeg output

        # Extract width and height of the frame
        img = cv2.imread(save_path)
        height, width = img.shape[:2]

        coco["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        # Extract annotations from xml
        for obj in frame.findall("object"):
            category_name = obj.attrib["objectType"]
            if category_name in category_id_map:
                category_id = category_id_map[category_name]
                category_id_map[category_name] = category_id
                coco["categories"].append(
                    {
                        "id": category_id,
                        "name": category_name,
                    }
                )
            else:
                category_id = category_id_map[category_name]

            contour = obj.find("contour").text.strip().split(",")

            segmentation = []

            for point in contour:
                x, y = map(float, point.split())
                segmentation.extend([x, y])

            xs = segmentation[::2]
            ys = segmentation[1::2]
            xmin = min(xs)
            ymin = min(ys)
            width = max(xs) - xmin
            height = max(ys) - ymin

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [],
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        image_id += 1

    return coco, annotation_id, image_id


def clean_annotations():
    check_ffmpeg()

    os.makedirs(output_dir, exist_ok=True)

    coco_labels = {"images": [], "annotations": []}

    category_id_map = get_all_categories()
    annotation_id = 0
    image_id = 0

    # Extract frames (and their annotations) for each video
    for video_path in mp4_dir.rglob("*.mp4"):
        video_name = video_path.stem
        print("Extracting frames from video: ", video_name)
        label_path = label_dir / f"{video_name}.xml"

        coco_dict, annotation_id, image_id = extract_keyframes(
            label_path, video_path, video_name, category_id_map, annotation_id, image_id
        )
        coco_labels["images"].extend(coco_dict["images"])
        coco_labels["annotations"].extend(coco_dict["annotations"])

    coco_labels["categories"] = [
        {"id": id, "name": name} for name, id in category_id_map.items()
    ]

    # write coco json
    with open(os.path.join(output_dir, "annotations_coco.json"), "w") as f:
        json.dump(coco_labels, f, indent=2)

    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=output_dir, annotations_path=annotations_path
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset classes: {dataset.classes}")


if __name__ == "__main__":
    extract_data()
    clean_annotations()

    # # Visualize data and save sample image
    image_sample = build_and_visualize_supervision_dataset_from_coco_dataset(output_dir, annotations_path)
    plt.imsave("data_preview/f4k_sample_image.png", image_sample)
