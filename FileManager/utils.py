import os
import cv2


def save(image_crops, mask_crops, image_idx, grid_size, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save image crops
    for idx, crop in enumerate(image_crops):
        image_filename = f"image_{image_idx}_grid_{grid_size}_{idx}.png"
        image_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(image_path, crop)

    # Save mask crops
    for idx, crop in enumerate(mask_crops):
        mask_filename = f"image_{image_idx}_grid_{grid_size}_{idx}_mask.png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, crop)

    return True