import cv2
import os

from Visualizer.utils import create_canvas, collect_crops_from_path


def display_grid_from_path(output_dir, image_idx, grid_size, mask=False, spacing=5):
    image_folder = os.path.join(output_dir, f"image_{image_idx}")

    crop_paths = collect_crops_from_path(image_folder, grid_size, image_idx)

    # Load images
    grid_cells = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in crop_paths]
    if not grid_cells or any(cell is None for cell in grid_cells):
        print("Some grid cells could not be loaded.")
        return

    target_h, target_w = grid_cells[0].shape

    # Create canvas`
    create_canvas(f'{grid_size}x{grid_size} Grid (Image {image_idx})', target_h, target_w, grid_size, grid_cells, spacing)