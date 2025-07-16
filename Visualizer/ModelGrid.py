import os
import numpy as np
import cv2

from Visualizer.utils import create_canvas, collect_crops_from_path


def display_predicted_grid_tf(output_dir, image_idx, grid_size, model, mask_threshold=0.5, spacing=5):
    image_folder = os.path.join(output_dir, f"image_{image_idx}")


    crop_paths = collect_crops_from_path(image_folder, grid_size, image_idx)

    # Load images
    grid_cells = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in crop_paths]
    if not grid_cells or any(cell is None for cell in grid_cells):
        print("Some grid cells could not be loaded.")
        return

    # Prepare crops for model: [N, H, W, 1], normalized to [0,1]
    crops_np = np.stack(grid_cells, axis=0).astype(np.float32) / 255.0
    crops_np = np.expand_dims(crops_np, axis=-1)  # [N, H, W, 1]

    # Predict masks (batch inference)
    preds = model.predict(crops_np)
    # If model output is [N, H, W, 1], squeeze last axis
    if preds.shape[-1] == 1:
        preds = np.squeeze(preds, axis=-1)  # [N, H, W]

    # Binarize masks
    masks = (preds > mask_threshold).astype(np.uint8) * 255  # [N, H, W]

    target_h, target_w = masks[0].shape

    create_canvas(f'U-Net Predictions: {grid_size}x{grid_size} Grid (Image {image_idx})', target_h, target_w, grid_size, grid_cells, spacing)