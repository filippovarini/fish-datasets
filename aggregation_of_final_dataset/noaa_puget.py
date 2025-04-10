from data_preview.visualise_noaa_puget import download_data, DATA_DIR, ANNOTATIONS_PATH
from aggregation_of_final_dataset.utils import compress_annotations_to_single_category

download_data(DATA_DIR)

# We compress all annotations into a single category: Fish
compressed_annotations_path = compress_annotations_to_single_category(ANNOTATIONS_PATH)

