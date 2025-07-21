import cv2
import os

def create_grid_cells(image, n):
    height, width = image.shape
    cell_height = height // n
    cell_width = width // n
    grid_cells = []

    for i in range(n):
        for j in range(n):
            y_start = i * cell_height
            x_start = j * cell_width
            y_end = (i + 1) * cell_height if i < n - 1 else height
            x_end = (j + 1) * cell_width if j < n - 1 else width
            cell = image[y_start:y_end, x_start:x_end]
            grid_cells.append(cell)

    return grid_cells


def extract_center_crops(image, crop_output):
    height, width = image.shape
    center_x, center_y = width // 2, height // 2
    crop_size = min(height, width, crop_output)
    y_start = center_y - crop_size // 2
    x_start = center_x - crop_size // 2
    return image[y_start:y_start + crop_size, x_start:x_start + crop_size]


def resize_cells(cells, crop_output):
    return [cv2.resize(cell, (crop_output, crop_output)) for cell in cells]


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
