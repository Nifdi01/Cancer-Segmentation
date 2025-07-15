from Visualizer.utils import create_canvas


def display_grid(grid_cells, grid_size, spacing=5):
    grid_cells = grid_cells[:grid_size * grid_size]
    target_h, target_w = grid_cells[0].shape
    print(target_h, target_w)
    # Create canvas
    create_canvas(f'{grid_size}x{grid_size} Grid with Spacing', target_h, target_w, grid_size, grid_cells, spacing)


