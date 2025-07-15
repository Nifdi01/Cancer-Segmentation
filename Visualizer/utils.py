import matplotlib.pyplot as plt
import cv2
import numpy as np


def create_canvas(title, target_h, target_w, grid_size, grid_cells, spacing=5):
    grid_height = grid_size * target_h + (grid_size - 1) * spacing
    grid_width = grid_size * target_w + (grid_size - 1) * spacing
    canvas = np.ones((grid_height, grid_width), dtype=np.uint8) * 255

    for idx, cell in enumerate(grid_cells):
        row = idx // grid_size
        col = idx % grid_size
        y = row * (target_h + spacing)
        x = col * (target_w + spacing)
        canvas[y:y + target_h, x:x + target_w] = cell

    # Convert to RGB for matplotlib
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas)
    plt.axis('off')
    plt.title(title)
    plt.show()