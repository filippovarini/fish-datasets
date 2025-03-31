import os
import pandas as pd
import supervision as sv
import cv2
import numpy as np
from tqdm import tqdm

def download_unzip(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.system(f'wget -O {output_path}.zip {url}')
    os.system(f'unzip {output_path}.zip -d {output_path}')

# Download data
download_url = 'https://fishnet-data.s3-us-west-1.amazonaws.com/foid_images_v100.zip'
folder = '../downloads'
dataset_name = 'fishnet'
local_path = f'{folder}/{dataset_name}'
download_unzip(download_url, local_path)

# Download annotations
annotation_folder = f'{folder}/{dataset_name}_annotations'
download_unzip('https://fishnet-data.s3-us-west-1.amazonaws.com/foid_labels_v100.zip', annotation_folder)

# Update annotations to match Supervision format
annotation_path = os.path.join(annotation_folder, 'foid_labels_v100.csv')
new_annotation_path = os.path.join(annotation_folder, 'foid_labels_v100_new.csv')

# Read and transform annotations
annotations_df = pd.read_csv(annotation_path)
annotations_df.rename(columns={
    'img_id': 'filename',
    'label_l1': 'class',
    'x_min': 'xmin',
    'x_max': 'xmax',
    'y_min': 'ymin',
    'y_max': 'ymax'
}, inplace=True)
annotations_df['filename'] = annotations_df['filename'].apply(lambda x: f'{x}.jpg')
# Set default image dimensions
annotations_df['height'] = 1920
annotations_df['width'] = 1080

# Save the transformed annotations
annotations_df.to_csv(new_annotation_path, index=False)

# Load the dataset using Supervision
dataset = sv.DetectionDataset.from_csv(
    images_directory=os.path.join(local_path, 'images'),
    annotations_path=new_annotation_path,
    images_extension='.jpg'
)

# Print dataset summary
print("\n" + "=" * 80)
print("IMAGE ANNOTATION SUMMARY".center(80))
print("=" * 80)
print(f"number of images         : {len(dataset)}")
print(f"folder image counts      :")
print(f"                         > images : {len(dataset)}")

# Get unique image sizes
image_sizes = set()
for _, image_info in dataset.images.items():
    image_sizes.add((image_info.width, image_info.height))
print(f"number of image sizes    : {len(image_sizes)}")
print(f"image_size               : {list(image_sizes)[0][0]} X {list(image_sizes)[0][1]}")

# Get class statistics
class_counts = {}
for _, annotation in dataset.annotations.items():
    for det in annotation:
        class_name = dataset.classes[det.class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

print(f"number of object classes : {len(class_counts)}")
print(f"object classes           : {' | '.join(class_counts.keys())}")
print(f"number of objects        : {sum(class_counts.values())}")
print(f"class object count       :")
for class_name, count in class_counts.items():
    print(f"                         > {class_name.ljust(25)} : {count}")
print("=" * 80 + "\n")

def visualize_sample(dataset, sample_idx=0):
    # Get a sample image and its annotations
    image_name = list(dataset.images.keys())[sample_idx]
    image_info = dataset.images[image_name]
    image_path = image_info.path
    annotations = dataset.annotations[image_name]
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create detections from annotations
    detections = sv.Detections(
        xyxy=np.array([ann.bbox.xyxy for ann in annotations]),
        class_id=np.array([ann.class_id for ann in annotations]),
        confidence=np.ones(len(annotations))
    )
    
    # Create annotator and annotate image
    box_annotator = sv.BoxAnnotator()
    labels = [dataset.classes[class_id] for class_id in detections.class_id]
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    
    # Display the image
    return annotated_image

# Visualize a few samples
for i in range(min(5, len(dataset))):
    annotated_image = visualize_sample(dataset, i)
    # In a Jupyter notebook, you would display this with:
    # plt.figure(figsize=(12, 8))
    # plt.imshow(annotated_image)
    # plt.axis('off')
    # plt.show()
    
    # For saving instead:
    output_dir = 'visualization_outputs'
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/sample_{i}.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# Split dataset into train and validation sets
train_ratio = 0.8
image_names = list(dataset.images.keys())
np.random.shuffle(image_names)
split_idx = int(len(image_names) * train_ratio)
train_names = image_names[:split_idx]
val_names = image_names[split_idx:]

# Create train and validation datasets
train_dataset = sv.DetectionDataset(
    images={name: dataset.images[name] for name in train_names},
    annotations={name: dataset.annotations[name] for name in train_names},
    classes=dataset.classes
)

val_dataset = sv.DetectionDataset(
    images={name: dataset.images[name] for name in val_names},
    annotations={name: dataset.annotations[name] for name in val_names},
    classes=dataset.classes
)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Export to COCO format for compatibility with various training frameworks
train_coco_path = os.path.join(folder, f"{dataset_name}_train.json")
val_coco_path = os.path.join(folder, f"{dataset_name}_val.json")

train_dataset.as_coco(train_coco_path)
val_dataset.as_coco(val_coco_path)

print(f"Exported training set to: {train_coco_path}")
print(f"Exported validation set to: {val_coco_path}")

