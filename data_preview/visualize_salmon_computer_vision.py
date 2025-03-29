"""

This notebook convert this dataset to COCO:
    
https://github.com/Salmon-Computer-Vision/salmon-computer-vision

The dataset includes two zipfiles that I *think* are mostly the same images, 
with different annotation formats.  It's not clear why two zipfiles were provided.
They also don't use *exactly* the same files, so it's hard to precisely compare them.

For example, the bytetrack-formatted filenames look like this:
    
    test/07-15-2020_18-48-46_m_left_bank_underwater/img1/000001.jpg

...and the YOLO-formatted files look like this:
    
    images/test/07-15-2020_18-48-46_m_left_bank_underwater_frame_000000.jpg
    
...but those appear to be the same image.  The conclusion of my exploration here
is that I can ignore the bytetrack-formatted data, and keep just "yolov6_salmon.tar.gz".

The zipfile also does not include a class list file, but I'm 90% sure this is the right 
class list file:
    
https://github.com/Salmon-Computer-Vision/salmon-computer-vision/blob/master/training/2023_combined_salmon.yaml

It doesn't matter a lot in our case, since we're reducing everything to just "fish", but I'm 
90% sure those are the classes used in the YOLO version of the datset.

"""

#%% Imports and constants

import os

dataset_root = r'i:\data\salmon'
bytetrack_root = os.path.join(dataset_root,'bytetrack_salmon/datasets/salmon')
yolo_root = os.path.join(dataset_root,'yolov6_salmon/export_yolov6_combined_bear_kitwanga_preprocess_yolo')

assert os.path.isdir(bytetrack_root)
assert os.path.isdir(yolo_root)


#%% List files in both versions of the dataset

from megadetector.utils.path_utils import recursive_file_list
bytetrack_files_relative = recursive_file_list(bytetrack_root,return_relative_paths=True)
yolo_files_relative = recursive_file_list(yolo_root,return_relative_paths=True)

print('Enumerated {} files in the bytetrack folder'.format(len(bytetrack_files_relative)))
print('Enumerated {} files in the yolo folder'.format(len(yolo_files_relative)))

"""
Enumerated 1415290 files in the bytetrack folder
Enumerated 2782343 files in the yolo folder
"""


#%% Find image files in both versions of the dataset

from megadetector.utils.path_utils import is_image_file
bytetrack_image_files = [fn for fn in bytetrack_files_relative if is_image_file(fn)]
yolo_image_files = [fn for fn in yolo_files_relative if is_image_file(fn)]

print('The bytetrack data contains {} images'.format(len(bytetrack_image_files)))
print('The yolo data contains {} images'.format(len(yolo_image_files)))

"""
The bytetrack data contains 1410268 images
The yolo data contains 1391168 images

I don't know what the discrepancy is, but it seems a lot easier to use the YOLO
data, and a visual inspection says that they're similar, so I'm going to ignore
the bytetrack data.
"""


#%% Make sure there is a YOLO annotation file for every image in the YOLO dataset

from tqdm import tqdm

"""

Sample image/label files:
    
'images/test/07-15-2020_18-48-46_m_left_bank_underwater_frame_000000.jpg'
'labels/test/07-15-2020_18-48-46_m_left_bank_underwater_frame_000000.txt'

"""

yolo_image_files = [fn for fn in yolo_files_relative if is_image_file(fn)]
yolo_text_files = [fn for fn in yolo_files_relative if fn.endswith('.txt')]

assert len(yolo_image_files) == len(yolo_text_files)

yolo_image_root_names = [os.path.splitext(fn.replace('images/',''))[0] \
                         for fn in yolo_image_files]
                                  
yolo_image_root_names_set = set(yolo_image_root_names)

for fn in tqdm(yolo_text_files):
    text_filename_root = os.path.splitext(fn.replace('labels/',''))[0]
    assert text_filename_root in yolo_image_root_names_set


#%% Convert to COCO

from megadetector.data_management.yolo_to_coco import yolo_to_coco
    
yolo_image_root = os.path.join(yolo_root,'images')    
yolo_label_root = os.path.join(yolo_root,'labels')
yolo_dataset_file = os.path.join(yolo_root,'2023_combined_salmon.yaml')
coco_dataset_file = os.path.join(yolo_root,'salmon_computer_vision_coco.json')

assert os.path.isdir(yolo_image_root) and os.path.isdir(yolo_label_root)
assert os.path.isfile(yolo_dataset_file)

input_folder = yolo_image_root
class_name_file = yolo_dataset_file
output_file = coco_dataset_file
empty_image_handling = 'no_annotations'
empty_image_category_name = 'empty'
error_image_handling = 'no_annotations'
allow_images_without_label_files = False
n_workers = 20
pool_type = 'thread'
recursive = True
exclude_string = None
include_string = None
overwrite_handling = 'overwrite'
label_folder = yolo_label_root

_ = yolo_to_coco(input_folder=input_folder,
                 class_name_file=class_name_file,
                 output_file=output_file,
                 empty_image_handling=empty_image_handling,
                 empty_image_category_name=empty_image_category_name,
                 error_image_handling=error_image_handling,
                 allow_images_without_label_files=allow_images_without_label_files,
                 n_workers=n_workers,
                 pool_type=pool_type,
                 recursive=recursive,
                 exclude_string=exclude_string,
                 include_string=include_string,
                 overwrite_handling=overwrite_handling,
                 label_folder=label_folder)


#%% Validate the output

from megadetector.data_management.databases.integrity_check_json_db import \
    IntegrityCheckOptions, integrity_check_json_db
    
options = IntegrityCheckOptions()
options.baseDir = input_folder
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
options.bRequireLocation = False
options.nThreads = 20
options.verbose = True
options.allowIntIDs = False

sorted_categories,d,error_info = integrity_check_json_db(coco_dataset_file,options)


#%% Preview

from megadetector.visualization.visualize_db import \
    DbVizOptions, visualize_db

preview_folder = r'g:\temp\salmon-preview'

options = DbVizOptions()

options.num_to_visualize = 1000
options.viz_size = (1280, -1)
options.htmlOptions['maxFiguresPerHtmlFile'] = 2000
options.parallelize_rendering = True
options.parallelize_rendering_with_threads = True
options.parallelize_rendering_n_cores = 25
options.show_full_paths = False
options.force_rendering = True

html_filename,_ = visualize_db(db_path=coco_dataset_file, 
                               output_dir=preview_folder,
                               image_base_dir=input_folder,
                               options=options)
