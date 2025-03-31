"""

This notebook assumes that you have downloaded the Coralscapes dataset:
    
https://josauder.github.io/coralscapes/    
...as a series of Parquet files.  It reads the Parquet files, extracts images to files,
converts the segmentation masks for the "fish" category to boxes, and writes annotations
to COCO format.

"""


#%% Imports and constants

import os
import io
import json

import pandas as pd
import numpy as np

# Not used directly, but requird for reading Parquet
import pyarrow # noqa

from PIL import Image
from tqdm import tqdm
from skimage import measure

from megadetector.utils.ct_utils import invert_dictionary
from megadetector.data_management.databases.integrity_check_json_db import \
    integrity_check_json_db, IntegrityCheckOptions
from megadetector.visualization.visualize_db import \
    DbVizOptions, visualize_db
from megadetector.utils.path_utils import open_file

input_folder = r'C:\git\coralscapes'

data_folder = os.path.join(input_folder,'data')
assert os.path.isdir(data_folder)

label_mapping_file = os.path.join(input_folder,'id2label.json')
with open(label_mapping_file,'r') as f:
    category_id_to_name = json.load(f)
assert isinstance(category_id_to_name,dict)
print('Read {} categories'.format(len(category_id_to_name)))

color_mapping_file = os.path.join(input_folder,'label2color.json')
with open(color_mapping_file,'r') as f:
    category_name_to_color = json.load(f)
assert isinstance(category_name_to_color,dict)
print('Read {} color mappings'.format(len(category_name_to_color)))

assert len(category_name_to_color) == len(category_id_to_name)

category_name_to_id = invert_dictionary(category_id_to_name)
fish_category_id = category_name_to_id['fish']
categories_to_include = [int(fish_category_id)]

output_image_folder = r'c:\temp\coralscapes-coco'
os.makedirs(output_image_folder,exist_ok=True)
output_coco_file = os.path.join(output_image_folder,'coralscapes.json')

output_preview_folder = r'c:\temp\coralscapes-coco-preview'
os.makedirs(output_preview_folder,exist_ok=True)


#%% List parquet files

parquet_files_relative = [fn for fn in os.listdir(data_folder) if fn.endswith('.parquet')]
print('Found {} parquet files'.format(len(parquet_files_relative)))


#%% Rendering and segmentation functions

def overlay_segmentation(img, label, category_id_to_name, category_name_to_color, alpha=0.5):
    """
    Overlay segmentation mask on an image with proper colors.
    
    Args:
        img: PIL Image of the original image
        label: PIL Image containing integer category IDs as pixel values
        category_id_to_name: Dictionary mapping category IDs to their names
        category_name_to_color: Dictionary mapping category names to RGB color lists
        alpha: Transparency of the overlay (0.0 to 1.0)
        
    Returns:
        PIL Image with segmentation overlay
    """
    
    # Convert images to numpy arrays for processing
    # img_array = np.array(img)
    label_array = np.array(label)
    
    # Create a blank RGB array for the mask
    mask_rgb = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
    
    # Fill in the RGB values for each category in the mask
    unique_ids = np.unique(label_array)
    
    for cat_id in unique_ids:
        
        # Skip background
        if cat_id == 0:
            continue
        
        cat_id_str = str(cat_id)
        
        # Get mask for this category
        if cat_id_str in category_id_to_name:            
            cat_name = category_id_to_name[cat_id_str]
            if cat_name in category_name_to_color:
                color = category_name_to_color[cat_name]
                # Apply color to all pixels with this category ID
                mask_rgb[label_array == cat_id] = color
    
    # Create the overlay by blending the original image with the colored mask
    overlay = Image.fromarray(mask_rgb)
    
    # Ensure img is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Blend the images
    blended = Image.blend(img, overlay, alpha)
    
    return blended

    
def get_bounding_boxes(mask_image, 
                       image_id, 
                       include_category_ids=None,
                       exclude_category_id=None):
    """
    Extract bounding boxes for all categories from a segmentation mask and return COCO-formatted
    annotations.
    
    This function finds all connected components (instances) of all categories in a 
    segmentation mask and generates bounding boxes for each instance in a simplified annotation
    format.
    
    Args:
        mask_image (PIL.Image): Segmentation mask as PIL Image with integer category IDs
        image_id (str): Image identifier to use for annotation ID strings
        include_category_ids (list, tuple, or set): list of category IDs for which we should generate boxes
        exclude_category_id (int, optional): Exclude one category (typically a background category)
        
    Returns:
        list: List of simplified annotation dictionaries containing:
            - id: Annotation identifier
            - image_id: Image identifier
            - category_id: The category ID of the instance
            - bbox: Bounding box as [x, y, width, height]            
    """
    
    # Convert PIL Image to numpy array
    mask = np.array(mask_image)
    
    # Find all unique category IDs in the mask (excluding 0 if it's background)
    unique_categories = np.unique(mask)
    if exclude_category_id is not None:
        if exclude_category_id in unique_categories:
            unique_categories = \
                unique_categories[unique_categories != exclude_category_id]
    
    annotations = []
    annotation_index = 0
    
    for category_id in unique_categories:
        
        if include_category_ids is not None and category_id not in include_category_ids:
            continue
        
        # Find connected components for this category
        mask_binary = (mask == category_id).astype(np.uint8)
        labeled_mask, num_labels = measure.label(mask_binary, return_num=True, connectivity=2)
        
        for label_id in range(1, num_labels + 1):
            # Get coordinates of pixels belonging to this instance
            y_indices, x_indices = np.where(labeled_mask == label_id)
            
            if len(y_indices) > 0:
                # Calculate bounding box [x, y, width, height]
                x_min, y_min = np.min(x_indices), np.min(y_indices)
                x_max, y_max = np.max(x_indices), np.max(y_indices)
                width, height = x_max - x_min + 1, y_max - y_min + 1
                
                annotation_id = image_id + '_ann_' + str(annotation_index).zfill(4)
                
                # Create simplified annotation
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': int(category_id),
                    'bbox': [int(x_min), int(y_min), int(width), int(height)]
                }
                annotations.append(annotation)
                annotation_index += 1
                
        # ...for each instance
        
    # ...for each category
    
    return annotations
            
# ...def get_bounding_boxes(...)


#%% Process Paruet files

output_dict = {}
output_dict['info'] = {}
output_dict['info']['version'] = '2025.03.28'
output_dict['info']['description'] = 'Coralscapes dataset, fish only, converted to boxes'

output_dict['images'] = []
output_dict['annotations'] = []
output_dict['categories'] = [{'id':int(fish_category_id),'name':'fish'}]

debug_max_file = None

# i_file = 0; fn_relative = parquet_files_relative[i_file]
for i_file,fn_relative in enumerate(parquet_files_relative):

    if (debug_max_file is not None) and (i_file > debug_max_file):
        break
    
    fn_abs = os.path.join(data_folder,fn_relative)
    df = pd.read_parquet(fn_abs)
    print('Read {} rows from {}'.format(len(df),fn_abs))
    
    # i_row = 100; row = df.iloc[i_row]
    for i_row,row in tqdm(df.iterrows(),total=len(df)):
                
        # Image and label filenames look like:
        #
        # 'site1_000001_016200_leftImg8bit.png'
        # 'site1_000001_016200_gtFine.png'
        
        image_fn_relative = row['image']['path']
        assert image_fn_relative.endswith('.png')
        assert '/' not in image_fn_relative and '\\' not in image_fn_relative
        
        # Read the image data
        image_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(image_bytes))
        
        # Read the label data
        label_fn_relative = row['label']['path']
        assert label_fn_relative.endswith('.png')
        assert '/' not in image_fn_relative and '\\' not in image_fn_relative            
        label_bytes = row['label']['bytes']
        label = Image.open(io.BytesIO(label_bytes))
        
        assert img.size == label.size
        
        image_tokens = image_fn_relative.split('_')
        site = image_tokens[0]
        assert site.startswith('site')
        
        label_tokens = label_fn_relative.split('_')
        for i_token in range(0,3):
            assert image_tokens[i_token] == label_tokens[i_token]
        
        # Scrap code to render segmentation masks during debugging
        if False:
            im_out = overlay_segmentation(img, label, category_id_to_name, category_name_to_color, alpha=0.5)
            target_width = 500
            im_out = im_out.resize((target_width, int(im_out.height * target_width / im_out.width)), Image.Resampling.LANCZOS)
        
        # Extract fish boxes
        boxes_this_image = get_bounding_boxes(mask_image=label,
                                              image_id=image_fn_relative,
                                              include_category_ids=categories_to_include,
                                              exclude_category_id=0)

        # Write the image out
        image_fn_output_abs = os.path.join(output_image_folder,image_fn_relative)
        img.save(image_fn_output_abs)

        im = {}
        im['file_name'] = image_fn_relative
        im['id'] = image_fn_relative
        im['width'] = img.size[0]
        im['height'] = img.size[1]
        im['location'] = site
        
        output_dict['images'].append(im)            
        output_dict['annotations'].extend(boxes_this_image)

    # ...for each image in this file

# ...for each parquet file


#%% Write the output COCO file

with open(output_coco_file,'w') as f:
    json.dump(output_dict,f,indent=1)
    
    
#%% Validate the output COCO file
    
options = IntegrityCheckOptions()
options.baseDir = output_image_folder    
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = True

_ = integrity_check_json_db(output_coco_file,options=options)


#%% Preview the output COCO file

options = DbVizOptions()
options.num_to_visualize = 2000
options.viz_size = (800, -1)
options.htmlOptions['maxFiguresPerHtmlFile'] = 2000
options.parallelize_rendering = True
options.parallelize_rendering_with_threads = True
options.parallelize_rendering_n_cores = 25

html_filename, _ = visualize_db(db_path=output_coco_file,
                                output_dir=output_preview_folder,
                                image_base_dir=output_image_folder)

open_file(html_filename)
