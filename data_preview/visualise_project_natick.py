# %%
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
import os
import xml.etree.ElementTree as ET

from data_preview.utils import visualize_supervision_dataset, download_file, extract_zip


DATASET_SHORTNAME = "project_natick"
DATA_DIR = Path("/mnt/data/dev/fish-datasets/data/raw") / DATASET_SHORTNAME
DATA_URL = "https://github.com/microsoft/Project_Natick_Analysis/releases/download/annotated_data/data_release.zip"


def add_extension_to_filename(directory, extension=".jpg"):
    """The XML files have a filename element that does not have an extension.
    This function adds the extension to the filename."""
    print(f"Checking and updating XML annotations in {directory}...")
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            filepath = Path(directory) / filename
            tree = ET.parse(filepath)
            root = tree.getroot()

            filename_element = root.find("filename")
            if filename_element is not None:
                if not filename_element.text.endswith(extension):
                    filename_element.text += extension
                    tree.write(filepath)
                    print(f"Updated {filename}")
                else:
                    print(f"{filename} already has extension, skipped.")


def download_data(data_dir: Path):
    data_dir.mkdir(exist_ok=True, parents=True)
    
    data_path = data_dir / "data_release.zip"
    
    if not data_dir.exists() or len(list(data_dir.glob("*"))) == 0:
        print("Downloading data...")
        download_file(DATA_URL, data_path)
        print("Extracting data...")
        extract_zip(data_path, data_dir)
    else:
        print("Data already downloaded and extracted")


def extract_example_image(images_path: Path, annotations_path: Path, dataset_shortname: str):
    dataset = sv.DetectionDataset.from_pascal_voc(
        images_directory_path=str(images_path),
        annotations_directory_path=str(annotations_path),
    )
    
    image_example = visualize_supervision_dataset(dataset)
    
    if image_example is not None:
        output_path = DATA_DIR / f"{dataset_shortname}_sample_image.png"
        plt.imsave(output_path, image_example)
        print(f"Sample image saved as {output_path}")
    else:
        print("No annotated images with detections found to save as example")


def main():
    download_data(DATA_DIR)
    
    dataset_path = DATA_DIR / "data_release" / "fish_arrow_worms_annotation"
    images_path = dataset_path / "JPEGImages"
    annotations_path = dataset_path / "Annotations"
    
    add_extension_to_filename(annotations_path)
    
    extract_example_image(images_path, annotations_path, DATASET_SHORTNAME)


if __name__ == "__main__":
    main()
