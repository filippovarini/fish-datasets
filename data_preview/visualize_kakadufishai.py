import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import random
from aggregation_of_final_dataset.settings import Settings
from data_preview.utils import save_sample_image, build_and_visualize_supervision_dataset_from_coco_dataset, download_file
import json
import subprocess
import sys



settings = Settings()
dataset_shortname = "kakadu"
data_dir = settings.raw_dir / dataset_shortname
data = 'https://zenodo.org/record/7250921/files/202210-KakaduFishAI-TrainingData.zip?download=1'
data_path = data_dir / "kakadu.zip"
images_path = data_dir
annotations_path = data_dir / "KakaduFishAI_boundingbox.json"

def data_extraction():
    try:
        subprocess.run(["unzip", str(data_path), "-d", str(data_dir)], check=True)
        data_path.unlink()
    except:
        print(f'\nError during extraction. Please install "unzip" tool (Linux/MacOS) or extract manually the data. \n\nIn case of manual Extraction, the result must be: data / raw / kakadu / (1.jpg | 2.jpg | 3.jpg | …)\n')
        sys.exit()


def download_data():
    # ## Download the Data
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True, parents=True)
    
    data_folder_content = list(data_dir.iterdir())

    match len(data_folder_content):

        case 0:
            download_file(data, data_path)
            data_extraction()

        case 1: # User has now installed unzip (assuming the only file present is the zip file)
            data_extraction()

        case _:
            print("Data already present!")
            if data_path.exists():
                data_path.unlink()



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