import json
import os
from PIL import Image, ImageDraw
import numpy as np

# Load the JSON data
json_file_path = r'Swat_new\images\_annotations.coco.json'  # Change this to your JSON file path
images_folder = r'Swat_new\images'                          # Path where your images are stored
masks_folder = r'Swat_new\masks'                   # Path to save the generated masks

# Create the output folder if it doesn't exist
os.makedirs(masks_folder, exist_ok=True)

# Load JSON data
with open(json_file_path) as f:
    data = json.load(f)

# Create a mapping of image_id to filename and annotations
image_id_to_annotations = {}
image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}

# Group annotations by image_id
for annotation in data['annotations']:
    image_id = annotation['image_id']
    if image_id not in image_id_to_annotations:
        image_id_to_annotations[image_id] = []
    image_id_to_annotations[image_id].append(annotation)

# Generate binary masks for each image
for image_id, annotations in image_id_to_annotations.items():
    # Get image file name using image_id
    image_file_name = image_id_to_filename.get(image_id)
    if not image_file_name:
        print(f"Image file for image_id {image_id} not found.")
        continue

    # Open the original image to get dimensions
    image_path = os.path.join(images_folder, image_file_name)
    with Image.open(image_path) as img:
        width, height = img.size

    # Create an empty (black) mask
    mask = Image.new('L', (width, height), 0)  # 'L' mode for (8-bit pixels, black and white)

    # Draw each segmentation onto the mask
    for annotation in annotations:
        segmentation = annotation['segmentation']
        for seg in segmentation:
            poly = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)

    # Convert mask to binary values (0, 255)
    binary_mask = np.array(mask) * 255
    binary_mask_img = Image.fromarray(binary_mask.astype(np.uint8))

    # Save mask with the same name as the original image but in the masks folder
    mask_file_name = os.path.splitext(image_file_name)[0] + '_mask.png'
    mask_file_path = os.path.join(masks_folder, mask_file_name)
    binary_mask_img.save(mask_file_path)

    print(f"Mask saved for image {image_file_name} at {mask_file_path}")
