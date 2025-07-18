import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_iou(pred_mask, true_mask):
    """Calculate Intersection over Union (IoU) between predicted and true masks."""
    pred_mask = (pred_mask > 0).astype(np.uint8)
    true_mask = (true_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0  # Both masks are empty

    return intersection / union


def reconstruct_from_grid(masks, grid_size, target_shape):
    """Reconstruct full-size prediction from grid of masks."""
    target_h, target_w = target_shape

    # Calculate dimensions for each grid cell
    cell_h = target_h // grid_size
    cell_w = target_w // grid_size

    # Create full prediction canvas
    full_prediction = np.zeros((target_h, target_w), dtype=np.uint8)

    for idx, mask in enumerate(masks):
        row = idx // grid_size
        col = idx % grid_size

        # Calculate position in full image
        y_start = row * cell_h
        y_end = min(y_start + cell_h, target_h)
        x_start = col * cell_w
        x_end = min(x_start + cell_w, target_w)

        # Resize mask to fit the cell dimensions
        mask_resized = cv2.resize(mask, (x_end - x_start, y_end - y_start))

        # Place in full prediction
        full_prediction[y_start:y_end, x_start:x_end] = mask_resized

    return full_prediction


def get_grid_file_paths(output_dir, image_idx, grid_size):
    """Collect crop and mask file paths for a grid."""
    image_folder = os.path.join(output_dir, f"image_{image_idx}")

    crop_paths = []
    mask_paths = []

    for idx in range(grid_size * grid_size):
        crop_filename = f"image_{image_idx}_grid_{grid_size}_{idx}.png"
        mask_filename = f"image_{image_idx}_grid_{grid_size}_{idx}_mask.png"

        crop_path = os.path.join(image_folder, crop_filename)
        mask_path = os.path.join(image_folder, mask_filename)

        crop_paths.append(crop_path)
        mask_paths.append(mask_path)

    return crop_paths, mask_paths


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


def load_images(image_paths, color_mode=cv2.IMREAD_GRAYSCALE):
    """Load images from paths and validate they loaded successfully."""
    images = [cv2.imread(path, color_mode) for path in image_paths]

    if not images or any(img is None for img in images):
        return None

    return images


def prepare_crops_for_model(grid_cells):
    """Prepare crop images for model inference."""
    crops_np = np.stack(grid_cells, axis=0).astype(np.float32) / 255.0
    crops_np = np.expand_dims(crops_np, axis=-1)  # [N, H, W, 1]
    return crops_np


def predict_masks(model, crops_np, mask_threshold=0.5):
    """Generate predictions from model and binarize them."""
    preds = model.predict(crops_np, verbose=0)

    # If model output is [N, H, W, 1], squeeze last axis
    if preds.shape[-1] == 1:
        preds = np.squeeze(preds, axis=-1)  # [N, H, W]

    # Binarize predictions
    pred_masks = (preds > mask_threshold).astype(np.uint8) * 255
    return pred_masks


def calculate_average_iou(pred_masks, gt_masks):
    """Calculate average IoU across all mask pairs."""
    total_iou = 0.0
    valid_cells = 0

    for i in range(len(pred_masks)):
        cell_iou = calculate_iou(pred_masks[i], gt_masks[i])
        total_iou += cell_iou
        valid_cells += 1

    return total_iou / valid_cells if valid_cells > 0 else 0.0


def plot_iou_scores(iou_scores, image_idx, best_grid_size):
    plt.figure(figsize=(10, 5))

    # Plot 1: IoU scores by grid size
    plt.subplot(1, 2, 1)
    grid_sizes = sorted(iou_scores.keys())
    ious = [iou_scores[gs] for gs in grid_sizes]
    plt.plot(grid_sizes, ious, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Grid Size')
    plt.ylabel('IoU Score')
    plt.title(f'IoU Score vs Grid Size for Image {image_idx}')
    plt.grid(True, alpha=0.3)
    plt.xticks(grid_sizes)

    # Highlight best grid size
    best_iou = iou_scores[best_grid_size]
    plt.plot(best_grid_size, best_iou, 'ro', markersize=12, label=f'Best: {best_grid_size}x{best_grid_size}')
    plt.legend()

    # Plot 2: Bar chart of IoU scores
    plt.subplot(1, 2, 2)
    colors = ['red' if gs == best_grid_size else 'blue' for gs in grid_sizes]
    plt.bar([f'{gs}x{gs}' for gs in grid_sizes], ious, color=colors, alpha=0.7)
    plt.ylabel('IoU Score')
    plt.title(f'IoU Scores by Grid Size for image {image_idx}')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()