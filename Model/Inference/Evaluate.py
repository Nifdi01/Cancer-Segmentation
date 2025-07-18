import os
import cv2
import numpy as np
from .utils import calculate_iou, get_grid_file_paths, validate_file_paths, load_images, prepare_crops_for_model, \
    predict_masks, calculate_average_iou


def predict_and_evaluate_grid_sizes(output_dir, image_idx, model,
                                    mask_threshold=0.5, grid_range=(4, 10)):
    """
    Evaluate different grid sizes using IoU and return the best one.

    Args:
        output_dir: Directory containing the cropped images (e.g., CROPS)
        image_idx: Index of the image to process
        model: Trained segmentation model
        mask_threshold: Threshold for binarizing predictions
        grid_range: Tuple of (min_grid_size, max_grid_size) inclusive

    Returns:
        best_grid_size: Grid size with highest IoU
        iou_scores: Dictionary of grid_size -> IoU score
    """
    min_grid, max_grid = grid_range
    iou_scores = {}

    # We'll collect ground truth masks for each grid size as we evaluate
    # Since the masks are grid-cell specific, we'll reconstruct them per grid size

    # Evaluate each grid size
    for grid_size in range(min_grid, max_grid + 1):
        iou_score = evaluate_grid_size(output_dir, image_idx, grid_size, model,
                                       mask_threshold)
        if iou_score is not None:
            iou_scores[grid_size] = iou_score
            print(f"Grid size {grid_size}x{grid_size}: IoU = {iou_score:.4f}")

    if not iou_scores:
        print("No valid IoU scores calculated.")
        return None, {}

    # Find best grid size
    best_grid_size = max(iou_scores.keys(), key=lambda x: iou_scores[x])
    best_iou = iou_scores[best_grid_size]

    print(f"\nBest grid size: {best_grid_size}x{best_grid_size} with IoU = {best_iou:.4f}")

    return best_grid_size, iou_scores


def evaluate_grid_size(output_dir, image_idx, grid_size, model, mask_threshold=0.5):
    """Evaluate a single grid size and return its IoU score."""

    # Get file paths
    crop_paths, mask_paths = get_grid_file_paths(output_dir, image_idx, grid_size)

    # Validate files exist
    valid_crop_paths, valid_mask_paths = validate_file_paths(crop_paths, mask_paths)
    if valid_crop_paths is None:
        return None

    # Load images
    grid_cells = load_images(valid_crop_paths)
    if grid_cells is None:
        print(f"Some grid cells could not be loaded for grid size {grid_size}.")
        return None

    gt_masks = load_images(valid_mask_paths)
    if gt_masks is None:
        print(f"Some ground truth masks could not be loaded for grid size {grid_size}.")
        return None

    # Prepare data for model
    crops_np = prepare_crops_for_model(grid_cells)

    # Generate predictions
    pred_masks = predict_masks(model, crops_np, mask_threshold)

    # Convert ground truth to numpy array
    gt_masks_np = np.stack(gt_masks, axis=0)

    # Calculate average IoU
    avg_iou = calculate_average_iou(pred_masks, gt_masks_np)

    return avg_iou
