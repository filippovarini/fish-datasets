from pathlib import Path
import os

from dotenv import load_dotenv
from roboflow import Roboflow, Project
from tqdm import tqdm

from aggregation_of_final_dataset.settings import Settings

settings = Settings()

load_dotenv()

DATASET_TO_UPLOAD = "fishclef"

def _get_roboflow_project():
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_KEY"))
    return rf.workspace("filippo-varini-public").project("foundational-fish-detector")


def _upload_split_to_roboflow(project: Project, dataset_name: str, split: str):
    print(f"Uploading {split} split to Roboflow")
    split_dir = Path(settings.processed_dir) / f"{dataset_name}_{split}"
    images_dir = split_dir / settings.images_folder_name
    annotation_path = split_dir / settings.coco_file_name

    print(split_dir)
    total_images = len(list(images_dir.glob("*")))

    for image_path in tqdm(list(images_dir.glob("*")), total=total_images):
        try:
            project.upload(
                image_path=str(image_path),
                annotation_path=str(annotation_path),
                split=split,
                tag_names=[dataset_name]
            )
        except Exception as e:
            print(f"⚠️⚠️ Error uploading {image_path}: {e}")


def main():
    print(f"Uploading {DATASET_TO_UPLOAD} to Roboflow")
    
    project = _get_roboflow_project()

    # Upload training data
    _upload_split_to_roboflow(project, DATASET_TO_UPLOAD, split="train")

    # Upload validation data
    _upload_split_to_roboflow(project, DATASET_TO_UPLOAD, split="val")


if __name__ == "__main__":
    main()
