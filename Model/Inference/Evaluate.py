import numpy as np

from utils import create_grid_cells, resize_cells, reconstruct_from_grid
from utils.image import load_images
from utils.file import collect_crops_from_path, validate_file_paths
from utils.model import prepare_crops_for_model, predict_masks, calculate_average_iou
import cv2


def predict_and_evaluate_grid_sizes(output_dir, image_idx, model,
                                    mask_threshold=0.5, grid_range=(4, 10)):
    min_grid, max_grid = grid_range
    iou_scores = {}


    # Evaluate each grid size
    for grid_size in range(min_grid, max_grid + 1):
        iou_score, _ = evaluate_grid_size(output_dir, image_idx, grid_size, model,
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

    return avg_iou, pred_masks


def voting_inference(image_path, grid_range, model, filter=None, mask_threshold=0.5, vote_threshold=4, crop_output=256):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    if filter is not None:
        image = filter.apply(image)

    original_shape = image.shape
    min_grid, max_grid = grid_range



    # Store binary predictions from each grid
    grid_predictions = []
    successful_grids = 0

    for i in range(min_grid, max_grid + 1):
        # Create grid cells
        grid_cells = create_grid_cells(image, i)

        # Resize all cells to same size
        resized_cells = resize_cells(grid_cells, crop_output=crop_output)

        # Prepare for model
        crops_np = prepare_crops_for_model(resized_cells)

        # Get predictions
        pred_masks = predict_masks(model, crops_np, mask_threshold)

        print(f"Grid {i}: {pred_masks.shape}")

        # Convert to list for reconstruction
        masks_list = [pred_masks[j] for j in range(len(pred_masks))]

        # Reconstruct to original image shape
        reconstructed = reconstruct_from_grid(masks_list, i, original_shape)

        # Binarize the reconstructed mask (ensure it's 0 or 1)
        binary_mask = (reconstructed > 127).astype(np.uint8)  # Assuming reconstructed is 0-255

        grid_predictions.append(binary_mask)
        successful_grids += 1

    if successful_grids == 0:
        print("No grids processed successfully!")
        return None

    # Convert list to array for easier processing
    grid_predictions = np.array(grid_predictions)  # Shape: (num_grids, height, width)

    # Apply voting mechanism
    vote_counts = np.sum(grid_predictions, axis=0)  # Count votes for each pixel
    final_mask = (vote_counts >= vote_threshold).astype(np.uint8)

    print(f"Voting statistics:")
    print(f"- Total grids: {successful_grids}")
    print(f"- Vote threshold: {vote_threshold}")
    print(f"- Pixels with max votes ({successful_grids}): {np.sum(vote_counts == successful_grids)}")
    print(f"- Pixels with >= {vote_threshold} votes: {np.sum(vote_counts >= vote_threshold)}")
    print(f"- Final positive pixels: {np.sum(final_mask)}")

    return final_mask


def heatmap_inference(image_path, grid_range, model, mask_threshold=0.5, crop_output=256, filter=None):
    """
    Perform ensemble inference on an image using multiple grid sizes.

    Args:
        image_path (str): Path to the input image
        grid_range (tuple): Range of grid sizes (min_grid, max_grid) inclusive
        model: Trained segmentation model
        mask_threshold (float): Threshold for binarizing predictions

    Returns:
        numpy.ndarray: Normalized ensemble prediction with same shape as input image
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    if filter is not None:
        image = filter.apply(image)

    original_shape = image.shape
    # crop_height = int(original_shape[0] * 0.85)  # Top 85% boundary
    # image[crop_height:, :] = 255.0
    min_grid, max_grid = grid_range

    final_mask = np.zeros(original_shape, dtype=np.float32)
    successful_grids = 0

    for i in range(min_grid, max_grid + 1):
        # Create grid cells
        grid_cells = create_grid_cells(image, i)

        # Resize all cells to same size
        resized_cells = resize_cells(grid_cells, crop_output=crop_output)

        # Prepare for model
        crops_np = prepare_crops_for_model(resized_cells)

        # Get predictions
        pred_masks = predict_masks(model, crops_np, mask_threshold)

        print(f"Grid {i}: {pred_masks.shape}")

        # Convert to list for reconstruction
        masks_list = [pred_masks[j] for j in range(len(pred_masks))]

        # Reconstruct to original image shape
        reconstructed = reconstruct_from_grid(masks_list, i, original_shape)
        final_mask += reconstructed.astype(np.float32)
        successful_grids += 1

    if successful_grids == 0:
        print("No grids processed successfully!")
        return None

    # Normalize by number of successful grids and convert to 0-1 range
    final_mask /= successful_grids
    final_mask /= 255.0

    return final_mask