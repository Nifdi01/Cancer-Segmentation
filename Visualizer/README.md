# Visualizer Module

The Visualizer module provides comprehensive visualization tools for medical image segmentation analysis, enabling interactive display of grid crops, model predictions, and performance metrics.

## Overview

This module offers specialized visualization capabilities for:
- **Grid Visualization**: Display organized grid crops with customizable spacing
- **Model Predictions**: Visualize U-Net segmentation results
- **Performance Analysis**: Interactive IoU score plotting and comparison
- **Interactive Display**: Matplotlib-based interactive visualizations

## Module Structure

```
Visualizer/
├── Grid.py        # Grid-based visualization tools
├── Model.py       # Model prediction visualization
└── __init__.py    # Module initialization
```

## Components

### Grid.py
Specialized tools for displaying grid-based image crops in organized layouts.

**Key Features:**
- Customizable grid spacing and layout
- Support for both images and masks
- Automatic canvas sizing and arrangement
- Path-based and array-based input support

### Model.py
Advanced visualization for model predictions and performance analysis.

**Key Features:**
- Best grid prediction display
- Batch inference visualization
- IoU score plotting with multiple chart types
- Interactive performance comparison tools

## Usage Examples

### Basic Grid Visualization

```python
from Visualizer import display_grid_from_path

# Display original image crops
display_grid_from_path(
    output_dir="CROPS",
    image_idx=5,
    grid_size=6,
    mask=False,    # Display images (not masks)
    spacing=5      # 5-pixel spacing between crops
)

# Display corresponding masks
display_grid_from_path(
    output_dir="CROPS",
    image_idx=5,
    grid_size=6,
    mask=True,     # Display masks
    spacing=5
)
```

### Model Prediction Visualization

```python
from Visualizer import display_best_grid_prediction
from Model.zoo import UNetModel1024

# Load trained model
model = UNetModel1024(weights="path/to/model.h5")
unet = model.load_model()

# Find and display best prediction
best_grid_size, iou_scores = display_best_grid_prediction(
    output_dir="CROPS",
    image_idx=10,
    model=unet,
    mask_threshold=0.7,
    spacing=5,
    grid_range=(4, 10)
)

print(f"Best grid size: {best_grid_size}x{best_grid_size}")
print("IoU scores by grid size:", iou_scores)
```

### Advanced Visualization Workflow

```python
from Visualizer import display_grid_from_path, display_predicted_grid_tf
from Model.zoo import UNetModel1024
import matplotlib.pyplot as plt

# Load model
model = UNetModel1024(weights="trained_model.h5")
unet = model.load_model()

# Set up comparison visualization
image_idx = 15
grid_size = 6

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle(f'Segmentation Analysis - Image {image_idx}', fontsize=16)

# Display original image crops
plt.subplot(2, 2, 1)
plt.title('Original Image Crops')
display_grid_from_path("CROPS", image_idx, grid_size, mask=False)

# Display ground truth masks
plt.subplot(2, 2, 2)
plt.title('Ground Truth Masks')
display_grid_from_path("CROPS", image_idx, grid_size, mask=True)

# Display model predictions
plt.subplot(2, 2, 3)
plt.title('Model Predictions')
display_predicted_grid_tf("CROPS", image_idx, grid_size, unet, mask_threshold=0.7)

# Display IoU comparison (would need separate implementation)
plt.subplot(2, 2, 4)
plt.title('Performance Metrics')
# Custom IoU visualization code here

plt.tight_layout()
plt.show()
```

### Batch Visualization

```python
from Visualizer import display_best_grid_prediction
import numpy as np

# Load model
model = UNetModel1024(weights="model.h5")
unet = model.load_model()

# Visualize multiple images
sample_images = np.random.choice(range(50), 5, replace=False)

results = []
for idx in sample_images:
    print(f"\n--- Processing Image {idx} ---")
    
    best_grid, iou_scores = display_best_grid_prediction(
        output_dir="CROPS",
        image_idx=idx,
        model=unet,
        mask_threshold=0.7,
        grid_range=(4, 8)
    )
    
    results.append({
        'image_idx': idx,
        'best_grid': best_grid,
        'best_iou': iou_scores[best_grid] if best_grid else 0,
        'all_ious': iou_scores
    })

# Summary statistics
best_grids = [r['best_grid'] for r in results if r['best_grid']]
best_ious = [r['best_iou'] for r in results if r['best_iou'] > 0]

print(f"\nSummary:")
print(f"Average best IoU: {np.mean(best_ious):.4f}")
print(f"Most common grid size: {max(set(best_grids), key=best_grids.count)}x{max(set(best_grids), key=best_grids.count)}")
```

## API Reference

### Grid Visualization Functions

#### `display_grid(grid_cells, grid_size, spacing=5)`
Display a grid of image arrays with customizable spacing.

**Parameters:**
- `grid_cells` (list): List of image arrays (numpy arrays)
- `grid_size` (int): Grid dimensions (n×n)
- `spacing` (int): Pixel spacing between grid cells

#### `display_grid_from_path(output_dir, image_idx, grid_size, mask=False, spacing=5)`
Display a grid of images loaded from file paths.

**Parameters:**
- `output_dir` (str): Directory containing cropped images
- `image_idx` (int): Index of the image to display
- `grid_size` (int): Grid dimensions (n×n)
- `mask` (bool): Whether to display masks (True) or images (False)
- `spacing` (int): Pixel spacing between grid cells

### Model Visualization Functions

#### `display_best_grid_prediction(output_dir, image_idx, model, mask_threshold=0.5, spacing=5, grid_range=(4, 10))`
Find and display the best grid prediction for an image.

**Parameters:**
- `output_dir` (str): Directory containing cropped images
- `image_idx` (int): Index of the image to analyze
- `model`: Trained segmentation model
- `mask_threshold` (float): Threshold for binary predictions
- `spacing` (int): Pixel spacing between predictions
- `grid_range` (tuple): Range of grid sizes to evaluate

**Returns:**
- `tuple`: (best_grid_size, iou_scores_dict)

#### `display_predicted_grid_tf(output_dir, image_idx, grid_size, model, mask_threshold=0.5, spacing=5)`
Display model predictions for a specific grid size.

**Parameters:**
- `output_dir` (str): Directory containing cropped images
- `image_idx` (int): Index of the image to analyze
- `grid_size` (int): Specific grid size to visualize
- `model`: Trained segmentation model
- `mask_threshold` (float): Threshold for binary predictions
- `spacing` (int): Pixel spacing between predictions

## Visualization Features

### Grid Layout System
- **Automatic Sizing**: Canvas dimensions calculated from grid size and spacing
- **Consistent Spacing**: Uniform gaps between image patches
- **Flexible Layout**: Supports various grid sizes (4x4 to 10x10+)
- **Quality Preservation**: Maintains original image quality

### Interactive Plotting
- **IoU Score Visualization**: Line plots and bar charts for performance analysis
- **Best Grid Highlighting**: Visual emphasis on optimal grid sizes
- **Multi-Metric Display**: Simultaneous display of various performance metrics
- **Customizable Styling**: Configurable colors, markers, and layouts

### Model Prediction Display
- **Real-Time Inference**: Live model predictions during visualization
- **Binary Threshold Control**: Adjustable threshold for segmentation masks
- **Batch Processing**: Efficient handling of multiple grid crops
- **Memory Management**: Optimized for large grid sizes

## Visualization Types

### 1. Grid Crop Display
```
┌─────┬─────┬─────┐
│ C₁  │ C₂  │ C₃  │
├─────┼─────┼─────┤
│ C₄  │ C₅  │ C₆  │
├─────┼─────┼─────┤
│ C₇  │ C₈  │ C₉  │
└─────┴─────┴─────┘
```

### 2. IoU Performance Charts
- **Line Plot**: IoU vs Grid Size progression
- **Bar Chart**: Comparative IoU scores across grid sizes
- **Highlighted Best**: Visual emphasis on optimal performance

### 3. Prediction Comparison
- **Side-by-Side**: Original, Ground Truth, Prediction
- **Overlay Visualization**: Prediction overlaid on original
- **Difference Maps**: Highlight prediction errors

## Customization Options

### Spacing and Layout
```python
# Tight spacing for detailed view
display_grid_from_path("CROPS", 5, 6, spacing=2)

# Wide spacing for clear separation
display_grid_from_path("CROPS", 5, 6, spacing=10)

# No spacing for seamless display
display_grid_from_path("CROPS", 5, 6, spacing=0)
```

### Threshold Adjustment
```python
# Conservative threshold (fewer false positives)
display_predicted_grid_tf("CROPS", 5, 6, model, mask_threshold=0.8)

# Liberal threshold (fewer false negatives)
display_predicted_grid_tf("CROPS", 5, 6, model, mask_threshold=0.3)
```

### Color and Style Customization
```python
import matplotlib.pyplot as plt

# Custom color scheme
plt.rcParams['image.cmap'] = 'viridis'  # Custom colormap
plt.rcParams['figure.facecolor'] = 'white'

# Custom figure sizing
plt.rcParams['figure.figsize'] = [12, 12]
```

## Important Notes

1. **Memory Usage**: Large grids may require substantial RAM for visualization
2. **Display Backend**: Requires matplotlib with appropriate backend for interactive display
3. **Image Format**: Expects grayscale images (single channel)
4. **File Dependencies**: Requires pre-processed crops from CropManager
5. **Model Compatibility**: Works with TensorFlow/Keras models outputting single-channel masks

## Troubleshooting

**Common Issues:**

- **Blank displays**: Check matplotlib backend configuration
- **Memory errors**: Reduce grid size or image resolution
- **Missing crops**: Ensure CropManager has processed the images
- **Incorrect dimensions**: Verify all crops have consistent dimensions
- **Display scaling**: Adjust figure size for better visibility

**Debug Tips:**

```python
# Check image dimensions
import cv2
img = cv2.imread("path/to/crop.png", cv2.IMREAD_GRAYSCALE)
print(f"Image shape: {img.shape}")

# Verify crop availability
import os
crop_path = "CROPS/image_5/image_5_grid_6_0.png"
print(f"File exists: {os.path.exists(crop_path)}")

# Test model prediction shape
prediction = model.predict(test_input)
print(f"Prediction shape: {prediction.shape}")
```

## Performance Tips

1. **Batch Size**: Use appropriate batch sizes for model inference
2. **Image Caching**: Cache loaded images for repeated visualization
3. **Display Resolution**: Adjust figure DPI for balance between quality and speed
4. **Memory Management**: Clear large arrays after visualization
5. **Interactive Mode**: Use `plt.ion()` for faster interactive plotting

## Integration

This module integrates seamlessly with:
- **CropManager**: Visualizes generated crops
- **Model**: Displays model predictions and evaluations
- **Jupyter Notebooks**: Perfect for interactive analysis
- **Research Workflows**: Ideal for paper figures and presentations
- **Quality Assurance**: Visual validation of processing pipelines

## Example Use Cases

### Research Analysis
```python
# Generate publication-quality figures
from Visualizer import display_best_grid_prediction
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(10, 8))

# Your visualization code here
display_best_grid_prediction("CROPS", 15, model, spacing=3)

plt.savefig('segmentation_results.pdf', dpi=300, bbox_inches='tight')
```

### Quality Control
```python
# Batch quality assessment
for i in range(0, 20, 5):
    print(f"Checking image {i}")
    display_grid_from_path("CROPS", i, 6, mask=False)
    display_grid_from_path("CROPS", i, 6, mask=True)
    input("Press Enter to continue...")  # Manual review
```

### Performance Comparison
```python
# Compare different thresholds
thresholds = [0.3, 0.5, 0.7, 0.9]
for threshold in thresholds:
    print(f"Threshold: {threshold}")
    display_predicted_grid_tf("CROPS", 10, 6, model, 
                            mask_threshold=threshold)
```

---

The Visualizer module provides essential tools for understanding and validating medical image segmentation results, enabling researchers and practitioners 
to gain insights into model performance and data quality through intuitive visual interfaces.
