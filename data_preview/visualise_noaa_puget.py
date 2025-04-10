import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import json

from utils import visualize_supervision_dataset, download_file, extract_zip


def clean_annotations(annotations_path: Path):
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



def extract_example_image(images_path: Path, annotations_path: Path, dataset_shortname: str):
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_path),
        annotations_path=str(annotations_path),
    )

    image_example = visualize_supervision_dataset(dataset)
    plt.imsave(f"{dataset_shortname}_sample_image.png", image_example)


def main():
    dataset_shortname = "noaa_puget"
    data_dir = Path("/mnt/data/dev/fish-datasets/data/raw") / dataset_shortname
    data_dir.mkdir(exist_ok=True, parents=True)
    
    data_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-images.zip"
    annotations_url = "https://storage.googleapis.com/public-datasets-lila/noaa-psnf/noaa_estuary_fish-annotations-2023.08.19.zip"

    data_path = data_dir / "images.zip"
    annotations_path = data_dir / "annotations.zip"
    
    download_file(data_url, data_path)
    download_file(annotations_url, annotations_path)
    
    extract_zip(data_path, data_dir)
    extract_zip(annotations_path, data_dir)
    
    annotations_path = data_dir / "noaa_estuary_fish-2023.08.19.json"
    images_path = data_dir / "JPEGImages"
    
    clean_annotations(annotations_path)
    
    extract_example_image(images_path, annotations_path, dataset_shortname)
    
    
if __name__ == "__main__":
    main()
