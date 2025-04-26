
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
from datetime import datetime
import random
import subprocess
import json
import sys


from fathomnet.api import boundingboxes, worms

from aggregation_of_final_dataset.settings import Settings

settings = Settings()

dataset_shortname = "fathomnet"
data_dir = settings.raw_dir / dataset_shortname
today = datetime.now().strftime("%Y.%m.%d")
annotations_path = data_dir / f"fathomnet-{today}.json"
images_path = data_dir / "images"



def download_data():

    if (data_dir.exists() and any(data_dir.iterdir())):
        print(f"Download folder ({data_dir}) already exists. Data will not be downloaded. Remove the folder to download data")
        return
    data_dir.mkdir(exist_ok=True, parents=True)
    print(f"Data will be downloaded in {data_dir}")

    root_concepts = ["Actinopterygii", "Sarcopterygii", "Chondrichthyes", "Myxini"]
    print(f"Root concepts: {', '.join(root_concepts)}\n")

    # Get all descendants of the root concepts
    fish_concepts = set(root_concepts)
    for rc in root_concepts:
        descendant_concepts = worms.get_descendants_names(rc)
        fish_concepts.update(descendant_concepts)
        print(f"Added {len(descendant_concepts)} descendants of {rc}")

    # Find annotated concepts in FathomNet
    fathomnet_concepts = set(boundingboxes.find_concepts())

    # Compute the intersection of fish concepts and FathomNet concepts
    fish_concepts_in_fathomnet = fish_concepts & fathomnet_concepts
    print(f"\nFound {len(fish_concepts_in_fathomnet)} fish concepts in FathomNet")

    # Write the fish concepts to a file
    fish_concepts_file = data_dir / "concepts.txt"
    with fish_concepts_file.open("w") as f:
        f.write("\n".join(sorted(fish_concepts_in_fathomnet)))
    print(f"Wrote selected concepts to {fish_concepts_file}")
    print(f"Annotations will be saved to {annotations_path}")
    print(f"Images will be downloaded to {images_path}")

    # Download images and annotations
    python_executable = Path(sys.executable)
    fathomnet_generate = Path(python_executable).parent / "fathomnet-generate"

    # Check that this next print shows the same location of the output of "which fathomnet-generate"
    print(f"Path to fathomnet-generate: {fathomnet_generate}")
    cmd = [
        str(fathomnet_generate),
        "-v",
        "--format", "coco",
        "--concepts-file", str(fish_concepts_file),
        "--output", str(data_dir),
        "--img-download", str(images_path)
    ]

    subprocess.run(cmd, check=True)
    subprocess.run(["mv", data_dir / "dataset.json", annotations_path], check=True)
    



def clean_annotations():
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


# ## Visualise
# To visualise we need to extract the frames from the video, therefore, pick only one video to analyse
def visualize():
    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=str(images_path),
        annotations_path=str(annotations_path),
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset classes: {dataset.classes}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    image_example = None

    annotated_images = []
    for _ in range(16):
        i = random.randint(0, len(dataset))
        
        _, image, annotations = dataset[i]

        labels = [dataset.classes[class_id] for class_id in annotations.class_id]

        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, annotations)
        annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
        annotated_images.append(annotated_image)
        
        if len(annotations) > 0:
            image_example = annotated_image
        
    sv.plot_images_grid(
        annotated_images,
        grid_size=(4, 4),
        titles=None,
        size=(20, 12),
        cmap="gray"
    )

    image_example = image_example[..., ::-1]  # BGR to RGB
    plt.imsave(f"data_preview/{dataset_shortname}_sample_image.png", image_example)

if __name__ == "__main__":
    download_data()
    clean_annotations()
    visualize()