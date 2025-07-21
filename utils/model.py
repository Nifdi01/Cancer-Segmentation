import numpy as np


def calculate_iou(pred_mask, true_mask):
    """Calculate Intersection over Union (IoU) between predicted and true masks."""
    pred_mask = (pred_mask > 0).astype(np.uint8)
    true_mask = (true_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0  # Both masks are empty

    return intersection / union


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


def prepare_crops_for_model(grid_cells):
    """Prepare crop images for model inference."""
    crops_np = np.stack(grid_cells, axis=0).astype(np.float32) / 255.0
    crops_np = np.expand_dims(crops_np, axis=-1)  # [N, H, W, 1]
    return crops_np