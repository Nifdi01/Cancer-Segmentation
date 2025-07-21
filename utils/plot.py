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
