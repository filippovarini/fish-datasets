from pathlib import Path
import json
from datetime import datetime
import subprocess

import supervision as sv
import matplotlib.pyplot as plt
from fathomnet.api import boundingboxes, worms

from data_preview.utils import (
    visualize_supervision_dataset,
    download_file,
    extract_downloaded_file,
    CompressionType,
)


DATASET_SHORTNAME = "fathomnet"
DATA_DIR = Path("/Volumes/G-DRIVE-ArmorATD/MFD/raw") / DATASET_SHORTNAME


def download_data(data_dir: Path):
    if data_dir.exists() and len(list(data_dir.glob("*.json"))) > 0:
        print(f"Dataset already exists in {data_dir}")
        return
    
    # Choose the root concepts for fish
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
    
    
    today = datetime.now().strftime("%Y.%m.%d")
    annotations_path = data_dir / f"fathomnet-{today}.json"
    images_path = data_dir / "images"
    print(f"Annotations will be saved to {annotations_path}")
    print(f"Images will be downloaded to {images_path}")
    
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True, parents=True)
    images_path.mkdir(exist_ok=True, parents=True)
    
    cmd = [
        "fathomnet-generate", "-v",
        "--format", "coco",
        "--concepts-file", str(fish_concepts_file),
        "--output", str(data_dir),
        "--img-download", str(images_path)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    return annotations_path, images_path


def main():
    annotations_path, images_path = download_data(DATA_DIR)
    # visualize_supervision_dataset(annotations_path, images_path)


# if __name__ == "__main__":
#     main()
    
    