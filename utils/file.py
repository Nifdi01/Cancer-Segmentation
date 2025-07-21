import cv2
import os

def collect_crops_from_path(image_folder, grid_size, image_idx):
    image_paths = []
    mask_paths = []
    for idx in range(grid_size * grid_size):  # +1 if center crop is saved
        image_filename = f"image_{image_idx}_grid_{grid_size}_{idx}.png"
        mask_filename = f"image_{image_idx}_grid_{grid_size}_{idx}_mask.png"

        image_path = os.path.join(image_folder, f"image_{image_idx}", image_filename)
        mask_path = os.path.join(image_folder, f"image_{image_idx}", mask_filename)

        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"Warning: {image_path} not found.")

        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            print(f"Warning: {mask_path} not found.")

    # Only return the first n*n paths (exclude center crop)
    image_paths = image_paths[:grid_size * grid_size]
    mask_paths = mask_paths[:grid_size * grid_size]

    return image_paths, mask_paths


def validate_file_paths(crop_paths, mask_paths):
    """Validate that all required files exist."""
    valid_crops = []
    valid_masks = []

    for crop_path, mask_path in zip(crop_paths, mask_paths):
        if os.path.exists(crop_path) and os.path.exists(mask_path):
            valid_crops.append(crop_path)
            valid_masks.append(mask_path)
        else:
            print(f"Warning: {crop_path} or {mask_path} not found.")
            return None, None

    return valid_crops, valid_masks
