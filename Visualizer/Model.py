import cv2
import os
import numpy as np

from utils.plot import plot_iou_scores, create_canvas
from Model.Inference.Evaluate import predict_and_evaluate_grid_sizes


def display_best_grid_prediction(output_dir, image_idx, model,
                                 mask_threshold=0.5, spacing=5, grid_range=(4, 10)):
    # Find best grid size
    best_grid_size, iou_scores = predict_and_evaluate_grid_sizes(
        output_dir, image_idx, model,
        mask_threshold, grid_range
    )

    if best_grid_size is None:
        print("Could not determine best grid size.")
        return None

    # Display the best prediction
    display_predicted_grid_tf(output_dir, image_idx, best_grid_size, model,
                              mask_threshold, spacing)

    # Plot IoU scores
    plot_iou_scores(iou_scores, image_idx, best_grid_size)

    return best_grid_size, iou_scores


def display_predicted_grid_tf(output_dir, image_idx, grid_size, model, mask_threshold=0.5, spacing=5):
    image_folder = os.path.join(output_dir, f"image_{image_idx}")

    # Collect all crop paths for this image and grid size
    crop_paths = []
    for idx in range(grid_size * grid_size + 1):  # +1 if you also saved the center crop
        filename = f"image_{image_idx}_grid_{grid_size}_{idx}.png"
        path = os.path.join(image_folder, filename)
        if os.path.exists(path):
            crop_paths.append(path)
        else:
            print(f"Warning: {path} not found.")

    # Only use the first n*n crops for the grid (ignore the center crop for display)
    crop_paths = crop_paths[:grid_size * grid_size]

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

    # Create canvas
    create_canvas(f'Best U-Net Predictions: {grid_size}x{grid_size} Grid (Image {image_idx})', target_h,
                  target_w, grid_size, masks, spacing)
