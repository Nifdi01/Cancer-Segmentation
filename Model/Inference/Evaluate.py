import numpy as np
from utils.image import load_images
from utils.file import collect_crops_from_path, validate_file_paths
from utils.model import prepare_crops_for_model, predict_masks, calculate_average_iou


def predict_and_evaluate_grid_sizes(output_dir, image_idx, model,
                                    mask_threshold=0.5, grid_range=(4, 10)):
    min_grid, max_grid = grid_range
    iou_scores = {}


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
    crop_paths, mask_paths = collect_crops_from_path(output_dir, grid_size, image_idx)

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
