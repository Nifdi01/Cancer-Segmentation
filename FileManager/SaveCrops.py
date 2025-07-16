import os
import cv2
from CropManager import crop
from .utils import save


def save_crops(input_dir, output_dir, grid_sizes=range(4, 11)):
    # Walk through all subdirectories and list all image files (excluding mask files)
    image_files = []
    for subdir, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.png') and not f.endswith('_mask.png'):
                image_files.append(os.path.join(subdir, f))

    for image_idx, image_path in enumerate(image_files):
        # Create a subdirectory for this image inside output_dir
        image_folder = os.path.join(output_dir, f"image_{image_idx}")
        os.makedirs(image_folder, exist_ok=True)

        # Construct mask path
        mask_path = image_path.replace('.png', '_mask.png')

        # Read mask if exists
        mask = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for grid_size in grid_sizes:
            # Crop image
            image_crops = crop(image_path, grid_size)
            # Crop mask if available
            mask_crops = []
            if mask is not None:
                mask_crops = crop(mask_path, grid_size)

            # Save crops in the image's subfolder
            save(image_crops, mask_crops, image_idx, grid_size, image_folder)

    return True