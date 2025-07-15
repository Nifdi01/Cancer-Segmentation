import cv2
import os

from Visualizer.utils import create_canvas


def display_grid_from_path(output_dir, image_idx, grid_size, mask=False, spacing=5):
    image_folder = os.path.join(output_dir, f"image_{image_idx}")

    # Collect all crop paths for this image and grid size
    crop_paths = []
    for idx in range(grid_size * grid_size + 1):  # +1 if you also saved the center crop
        if mask:
            filename = f"image_{image_idx}_grid_{grid_size}_{idx}_mask.png"
        else:
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

    target_h, target_w = grid_cells[0].shape

    # Create canvas
    create_canvas(f'{grid_size}x{grid_size} Grid (Image {image_idx})', target_h, target_w, grid_size, grid_cells, spacing)