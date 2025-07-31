from utils.plot import create_canvas
import cv2
from utils.file import collect_crops_from_path

def display_grid(grid_cells, grid_size, spacing=5):
    grid_cells = grid_cells[:grid_size * grid_size]
    target_h, target_w = grid_cells[0].shape
    print(target_h, target_w)
    # Create canvas
    create_canvas(f'{grid_size}x{grid_size} Grid with Spacing', target_h, target_w, grid_size, grid_cells, spacing)


def display_grid_from_path(output_dir, image_idx, grid_size, mask=False, spacing=5):

    crop_paths, mask_paths = collect_crops_from_path(output_dir, grid_size, image_idx)

    # Load images
    if mask:
        grid_cells = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]
    else:
        grid_cells = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in crop_paths]

    if not grid_cells or any(cell is None for cell in grid_cells):
        print("Some grid cells could not be loaded.")
        return

    target_h, target_w = grid_cells[0].shape

    # Create canvas`
    create_canvas(f'{grid_size}x{grid_size} Grid (Image {image_idx})', target_h, target_w, grid_size, grid_cells, spacing)

    return grid_cells