from pathlib import Path
import json
import shutil
import traceback

from tqdm import tqdm

from aggregation_of_final_dataset.settings import Settings

settings = Settings()


def main():
    final_dataset_dir = settings.processed_dir
    
    merged_dataset_dir = settings.base_dir / "community_fish_ai"
    merged_dataset_dir.mkdir(parents=True, exist_ok=True)
    merged_images_dir = merged_dataset_dir / settings.images_folder_name
    merged_images_dir.mkdir(parents=True, exist_ok=True)
    merged_annotations_file = merged_dataset_dir / settings.coco_file_name

    # Initialize the merged COCO dataset structure
    merged_coco = {
        "images": [],
        "annotations": [],
        "categories": settings.coco_categories
    }
    
    annotation_id_counter = 1
    image_id_old_to_new_map = {}

    for dataset in final_dataset_dir.glob("*"):
        if dataset.is_dir():
            is_train_dataset = dataset.name.endswith(settings.train_dataset_suffix)
            
            print(f"Processing {dataset.name} dataset. Is train: {is_train_dataset}...")
            
            # Read the COCO annotations file
            coco_file = dataset / settings.coco_file_name
            if not coco_file.exists():
                raise FileNotFoundError(f"No COCO file found in {dataset.name}")
                
                
            with open(coco_file, "r") as f:
                dataset_coco = json.load(f)
            
            # Process images
            for image in tqdm(dataset_coco["images"], desc="Processing images", total=len(dataset_coco["images"])):
                try:
                    # Copy image to merged directory
                    file_name = Path(image["file_name"]).name
                    src_image = dataset / settings.images_folder_name / file_name
                    dst_image = merged_images_dir / file_name
                    shutil.copy2(src_image, dst_image)
                    
                    # Update image info
                    old_image_id = image["id"]
                    image["id"] = len(image_id_old_to_new_map) + 1
                    image["is_train"] = is_train_dataset
                    merged_coco["images"].append(image)
                    image_id_old_to_new_map[old_image_id] = image["id"]
                except Exception as e:
                    print(f"Error processing image {image['file_name']}: {e}")
                    traceback.print_exc()
            
            # Process annotations
            for annotation in tqdm(dataset_coco["annotations"], desc="Processing annotations", total=len(dataset_coco["annotations"])):
                try:
                    annotation["id"] = annotation_id_counter
                    annotation["image_id"] = image_id_old_to_new_map[annotation["image_id"]]
                    merged_coco["annotations"].append(annotation)
                    annotation_id_counter += 1
                except Exception as e:
                    print(f"Error processing annotation {annotation['id']}: {e}")

    # Save the merged COCO annotations
    with open(merged_annotations_file, "w") as f:
        json.dump(merged_coco, f, indent=2)
    
    print(f"Successfully merged {len(merged_coco['images'])} images and {len(merged_coco['annotations'])} annotations")
    print(f"Merged dataset saved to {merged_dataset_dir}")


if __name__ == "__main__":
    main()
