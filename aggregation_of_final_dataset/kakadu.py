
from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import (
    
    compress_annotations_to_single_category,
)


settings = Settings()

DATASET_SHORTNAME = "kakadu"
annotations_path = settings.intermediate_dir / DATASET_SHORTNAME / "annotations_coco.json"

processing_dir = settings.intermediate_dir / DATASET_SHORTNAME



def main():



    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    # Keep all categories as all are fish
    categories_filter = None
    compress_annotations_to_single_category(
        annotations_path, categories_filter, compressed_annotations_path
    )





if __name__=="__main__":
    main()