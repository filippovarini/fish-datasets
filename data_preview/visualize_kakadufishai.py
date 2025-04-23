import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import random
from aggregation_of_final_dataset.settings import Settings
from data_preview.utils import save_sample_image, build_and_visualize_supervision_dataset_from_coco_dataset
import json
import subprocess



settings = Settings()
dataset_shortname = "kakadu"
data_dir = settings.raw_dir / dataset_shortname
data = 'https://zenodo.org/record/7250921/files/202210-KakaduFishAI-TrainingData.zip?download=1'
data_path = data_dir / "kakadu_data.zip"
images_path = data_dir
annotations_path = data_dir / "KakaduFishAI_boundingbox.json"


def download_data():
    # ## Download the Data
    # - If you want to use the `unzip` command you might need to install it. 
    # On linux, run `sudo apt-get install unzip`
    data_dir.mkdir(exist_ok=True, parents=True)

    subprocess.run(["curl", "-L", data, "-o", str(data_path)], check=True)
    subprocess.run(["unzip", str(data_path), "-d", str(data_dir)], check=True)
    subprocess.run(["rm", str(data_path)], check=True)




def clean_annotations():
    # ### Clean the annotations
    # Turn into COCO format readable by `supervision` library, for easy visualization and conversion to other formats.
    # - annotations.json only contains annotations for images with at least one bounding box

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

if __name__ == "__main__":
    download_data()
    clean_annotations()

    # Visualize data and save sample image
    image_sample = build_and_visualize_supervision_dataset_from_coco_dataset(
        images_path, 
        annotations_path)
    save_sample_image(dataset_shortname, image_sample)