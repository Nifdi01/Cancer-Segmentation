import cv2
from crop_center import extract_center_crops

def crop(image_path, n, crop_output=256):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image is not found or path is incorrect")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    cell_height = height // n
    cell_width = width // n

    grid_cells = []

    for i in range(n):
        for j in range(n):
            y_start = i * cell_height
            x_start = j * cell_width
            y_end = (i+1) * cell_height if i < n - 1 else height
            x_end = (j+1) * cell_width if j < n - 1 else width
            cell = image[y_start:y_end, x_start:x_end]
            grid_cells.append(cell)

    grid_cells.append(extract_center_crops(image, crop_output))

    # Resize all cells to the same size
    grid_cells = [
        cv2.resize(cell, (crop_output, crop_output)) for cell in grid_cells
    ]
    return grid_cells
