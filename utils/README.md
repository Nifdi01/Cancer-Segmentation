# Utils Module

The Utils module provides essential utility functions and helper tools that power the entire Cancer Segmentation framework. These utilities handle core operations like image processing, data manipulation, file operations, model utilities, and visualization helpers.

## Overview

This module contains foundational utilities organized into specialized submodules:
- **Image Processing**: Loading, validation, and encoding operations
- **Data Manipulation**: Dataset splitting, sampling, and transformation
- **File Operations**: Path handling, validation, and batch file operations
- **Model Utilities**: Inference helpers, metrics calculation, and preprocessing
- **Plotting Tools**: Visualization helpers and canvas creation
- **Cropping Operations**: Grid generation, resizing, and reconstruction

## Module Structure

```
utils/
├── image.py     # Image loading, validation, and encoding utilities
├── data.py      # Data manipulation and dataset operations
├── file.py      # File system operations and path handling
├── model.py     # Model inference and evaluation utilities
├── plot.py      # Visualization and plotting helpers
├── crop.py      # Image cropping and grid operations
└── __init__.py  # Module initialization with all exports
```

## Components

### image.py
Core image processing utilities for medical imaging workflows.

**Key Functions:**
- `load_image()`: PIL-based image loading with mode conversion
- `load_images()`: Batch image loading with validation
- `is_valid_image_file()`: Image file format validation
- `get_mask_path()`: Automatic mask path generation
- `encode_image_to_bytes()`: PNG encoding for TensorFlow records

### data.py
Data manipulation and dataset management utilities.

**Key Functions:**
- `split_dataset()`: Train/test splitting with validation
- `sample_distribution()`: Class-balanced sampling
- `iou_to_dataframe()`: Convert analysis results to structured data

### file.py
File system operations and path management utilities.

**Key Functions:**
- `collect_crops_from_path()`: Batch crop path collection
- `validate_file_paths()`: File existence validation

### model.py
Model inference and evaluation utilities.

**Key Functions:**
- `calculate_iou()`: IoU metric computation
- `predict_masks()`: Batch mask prediction with thresholding
- `calculate_average_iou()`: Multi-mask IoU averaging
- `prepare_crops_for_model()`: Input preprocessing for inference

### plot.py
Visualization and plotting utilities.

**Key Functions:**
- `create_canvas()`: Grid-based image canvas creation
- `plot_iou_scores()`: Performance visualization charts

### crop.py
Image cropping and grid manipulation utilities.

**Key Functions:**
- `create_grid_cells()`: Grid-based image division
- `extract_center_crops()`: Center-focused crop extraction
- `resize_cells()`: Batch image resizing
- `save()`: Structured crop saving
- `reconstruct_from_grid()`: Full-image reconstruction from grids

## Usage Examples

### Image Processing

```python
from utils.image import load_image, load_images, encode_image_to_bytes

# Load single image
image = load_image("path/to/image.png", mode="L")  # Grayscale
rgb_image = load_image("path/to/image.png", mode="RGB")  # Color

# Batch load with validation
image_paths = ["img1.png", "img2.png", "img3.png"]
images = load_images(image_paths)

if images is None:
    print("Some images failed to load")
else:
    print(f"Loaded {len(images)} images successfully")

# Encode for TensorFlow records
image_bytes = encode_image_to_bytes(image)
```

### Data Manipulation

```python
from utils.data import split_dataset, sample_distribution
import pandas as pd

# Create sample dataset
df = pd.DataFrame({
    'image_path': ['img1.png', 'img2.png', 'img3.png', 'img4.png'],
    'mask_path': ['mask1.png', 'mask2.png', 'mask3.png', 'mask4.png'],
    'class': ['positive', 'negative', 'positive', 'negative']
})

# Split dataset
train_df, test_df = split_dataset(df, train_ratio=0.8, test_ratio=0.2, seed=42)

# Balance class distribution
balanced_df = sample_distribution(
    df, 
    class_column='class',
    proportions={'positive': 0.6, 'negative': 0.4},
    random_state=42
)

print(f"Original: {len(df)}, Balanced: {len(balanced_df)}")
```

### Model Utilities

```python
from utils.model import calculate_iou, predict_masks, prepare_crops_for_model
import numpy as np

# Prepare data for model
grid_cells = [np.random.rand(256, 256) for _ in range(16)]  # Example crops
model_input = prepare_crops_for_model(grid_cells)
print(f"Model input shape: {model_input.shape}")

# Generate predictions (assuming you have a trained model)
# predictions = predict_masks(model, model_input, mask_threshold=0.7)

# Calculate IoU between predictions and ground truth
pred_mask = np.random.rand(256, 256) > 0.5  # Example prediction
gt_mask = np.random.rand(256, 256) > 0.5    # Example ground truth
iou_score = calculate_iou(pred_mask, gt_mask)
print(f"IoU Score: {iou_score:.4f}")
```

### File Operations

```python
from utils.file import collect_crops_from_path, validate_file_paths

# Collect crop paths for a specific image and grid size
crop_paths, mask_paths = collect_crops_from_path(
    image_folder="CROPS",
    grid_size=6,
    image_idx=5
)

print(f"Found {len(crop_paths)} crop files")
print(f"Found {len(mask_paths)} mask files")

# Validate all files exist
valid_crops, valid_masks = validate_file_paths(crop_paths, mask_paths)

if valid_crops is None:
    print("Some files are missing!")
else:
    print(f"All {len(valid_crops)} files validated successfully")
```

### Cropping Operations

```python
from utils.crop import create_grid_cells, extract_center_crops, resize_cells
import cv2

# Load image
image = cv2.imread("sample_image.png", cv2.IMREAD_GRAYSCALE)

# Create grid cells
grid_size = 6
grid_cells = create_grid_cells(image, grid_size)
print(f"Created {len(grid_cells)} grid cells from {grid_size}x{grid_size} grid")

# Extract center crop
center_crop = extract_center_crops(image, crop_output=256)

# Combine grid cells with center crop
all_crops = grid_cells + [center_crop]

# Resize all to consistent size
resized_crops = resize_cells(all_crops, crop_output=256)
print(f"Resized {len(resized_crops)} crops to 256x256")

# Reconstruct full image from grid (example with masks)
masks = [np.random.rand(256, 256) > 0.5 for _ in range(grid_size * grid_size)]
reconstructed = reconstruct_from_grid(masks, grid_size, target_shape=image.shape)
print(f"Reconstructed image shape: {reconstructed.shape}")
```

### Visualization

```python
from utils.plot import create_canvas, plot_iou_scores
import numpy as np

# Create sample grid for visualization
sample_cells = [np.random.rand(64, 64) * 255 for _ in range(16)]
create_canvas(
    title="Sample 4x4 Grid",
    target_h=64,
    target_w=64,
    grid_size=4,
    grid_cells=sample_cells,
    spacing=5
)

# Plot IoU scores
iou_scores = {4: 0.75, 5: 0.82, 6: 0.89, 7: 0.85, 8: 0.78}
plot_iou_scores(iou_scores, image_idx=10, best_grid_size=6)
```

## API Reference

### Image Utilities (`image.py`)

#### `load_image(image_path, mode="L")`
Load and convert image using PIL.

**Parameters:**
- `image_path` (str): Path to image file
- `mode` (str): PIL image mode ("L" for grayscale, "RGB" for color)

**Returns:**
- `PIL.Image`: Loaded and converted image

#### `load_images(image_paths, mode=cv2.IMREAD_GRAYSCALE)`
Batch load images using OpenCV.

**Parameters:**
- `image_paths` (list): List of image file paths
- `mode` (int): OpenCV imread mode

**Returns:**
- `list` or `None`: List of loaded images or None if any failed

#### `is_valid_image_file(filename)`
Check if file is a valid PNG image (not a mask).

**Parameters:**
- `filename` (str): Filename to validate

**Returns:**
- `bool`: True if valid image file

#### `get_mask_path(image_path)`
Generate corresponding mask path from image path.

**Parameters:**
- `image_path` (str): Original image path

**Returns:**
- `str`: Corresponding mask path

#### `encode_image_to_bytes(img)`
Encode PIL image to PNG bytes for TensorFlow records.

**Parameters:**
- `img` (PIL.Image): PIL image object

**Returns:**
- `bytes`: PNG-encoded image bytes

### Data Utilities (`data.py`)

#### `split_dataset(df, train_ratio=0.8, test_ratio=0.2, seed=42)`
Split DataFrame into training and testing sets.

**Parameters:**
- `df` (pandas.DataFrame): Input dataset
- `train_ratio` (float): Training set proportion
- `test_ratio` (float): Testing set proportion
- `seed` (int): Random seed for reproducibility

**Returns:**
- `tuple`: (train_df, test_df)

#### `sample_distribution(df, class_column, proportions, random_state=None)`
Sample DataFrame to achieve specified class proportions.

**Parameters:**
- `df` (pandas.DataFrame): Input dataset
- `class_column` (str): Column name containing class labels
- `proportions` (dict): Desired class proportions
- `random_state` (int, optional): Random seed

**Returns:**
- `pandas.DataFrame`: Sampled dataset

#### `iou_to_dataframe(results, index_range)`
Convert analysis results to structured DataFrame.

**Parameters:**
- `results` (dict): Results from analyze_iou_across_images()
- `index_range` (list): Original image indices

**Returns:**
- `pandas.DataFrame`: Detailed results with IoU scores

### File Utilities (`file.py`)

#### `collect_crops_from_path(image_folder, grid_size, image_idx)`
Collect crop file paths for specific image and grid size.

**Parameters:**
- `image_folder` (str): Root directory containing crops
- `grid_size` (int): Grid dimensions
- `image_idx` (int): Image index

**Returns:**
- `tuple`: (image_paths, mask_paths)

#### `validate_file_paths(crop_paths, mask_paths)`
Validate that all specified files exist.

**Parameters:**
- `crop_paths` (list): List of crop file paths
- `mask_paths` (list): List of mask file paths

**Returns:**
- `tuple`: (valid_crops, valid_masks) or (None, None) if validation fails

### Model Utilities (`model.py`)

#### `calculate_iou(pred_mask, true_mask)`
Calculate Intersection over Union between masks.

**Parameters:**
- `pred_mask` (numpy.ndarray): Predicted binary mask
- `true_mask` (numpy.ndarray): Ground truth binary mask

**Returns:**
- `float`: IoU score (0.0 to 1.0)

#### `predict_masks(model, crops_np, mask_threshold=0.5)`
Generate binary predictions from model output.

**Parameters:**
- `model`: Trained segmentation model
- `crops_np` (numpy.ndarray): Preprocessed crop array
- `mask_threshold` (float): Binarization threshold

**Returns:**
- `numpy.ndarray`: Binary prediction masks

#### `calculate_average_iou(pred_masks, gt_masks)`
Calculate average IoU across multiple mask pairs.

**Parameters:**
- `pred_masks` (numpy.ndarray): Array of predicted masks
- `gt_masks` (numpy.ndarray): Array of ground truth masks

**Returns:**
- `float`: Average IoU score

#### `prepare_crops_for_model(grid_cells)`
Preprocess crop images for model inference.

**Parameters:**
- `grid_cells` (list): List of crop image arrays

**Returns:**
- `numpy.ndarray`: Preprocessed array ready for model input

### Cropping Utilities (`crop.py`)

#### `create_grid_cells(image, n)`
Divide image into n×n grid cells.

**Parameters:**
- `image` (numpy.ndarray): Input image array
- `n` (int): Grid dimensions

**Returns:**
- `list`: List of grid cell arrays

#### `extract_center_crops(image, crop_output)`
Extract center crop from image.

**Parameters:**
- `image` (numpy.ndarray): Input image array
- `crop_output` (int): Desired crop size

**Returns:**
- `numpy.ndarray`: Center crop array

#### `resize_cells(cells, crop_output)`
Resize all cells to consistent dimensions.

**Parameters:**
- `cells` (list): List of image arrays
- `crop_output` (int): Target size

**Returns:**
- `list`: List of resized image arrays

#### `save(image_crops, mask_crops, image_idx, grid_size, output_dir)`
Save crops with structured naming convention.

**Parameters:**
- `image_crops` (list): List of image crop arrays
- `mask_crops` (list): List of mask crop arrays
- `image_idx` (int): Image index for naming
- `grid_size` (int): Grid size for naming
- `output_dir` (str): Output directory

**Returns:**
- `bool`: True if successful

#### `reconstruct_from_grid(masks, grid_size, target_shape)`
Reconstruct full-size image from grid of masks.

**Parameters:**
- `masks` (list): List of mask arrays
- `grid_size` (int): Original grid dimensions
- `target_shape` (tuple): Target reconstruction shape

**Returns:**
- `numpy.ndarray`: Reconstructed full-size mask

### Plotting Utilities (`plot.py`)

#### `create_canvas(title, target_h, target_w, grid_size, grid_cells, spacing=5)`
Create organized grid canvas for visualization.

**Parameters:**
- `title` (str): Canvas title
- `target_h` (int): Individual cell height
- `target_w` (int): Individual cell width
- `grid_size` (int): Grid dimensions
- `grid_cells` (list): List of image arrays to display
- `spacing` (int): Pixel spacing between cells

#### `plot_iou_scores(iou_scores, image_idx, best_grid_size)`
Create comprehensive IoU score visualization.

**Parameters:**
- `iou_scores` (dict): Dictionary of grid_size: iou_score pairs
- `image_idx` (int): Image index for labeling
- `best_grid_size` (int): Best performing grid size to highlight

## Design Principles

### Modularity
Each utility module handles a specific domain of functionality, enabling:
- **Clear Separation**: Distinct responsibilities for each module
- **Easy Testing**: Individual functions can be tested in isolation
- **Flexible Usage**: Import only needed functionality
- **Maintainability**: Changes isolated to specific domains

### Error Handling
Robust error handling throughout:
- **Graceful Failures**: Functions return None or meaningful defaults on errors
- **Informative Messages**: Clear error descriptions for debugging
- **Validation**: Input validation to prevent downstream errors
- **Logging**: Optional verbose output for monitoring

### Performance Optimization
Optimized for medical imaging workflows:
- **Batch Operations**: Efficient processing of multiple items
- **Memory Management**: Careful handling of large image arrays
- **Caching**: Where appropriate, results are cached for reuse
- **Vectorization**: NumPy operations for computational efficiency

## Important Notes

1. **Image Formats**: Primarily designed for PNG format medical images
2. **Memory Usage**: Large images and grids require substantial RAM
3. **Path Conventions**: Follows specific naming conventions for masks (`*_mask.png`)
4. **Dependencies**: Requires OpenCV, PIL, NumPy, pandas, matplotlib
5. **Thread Safety**: Functions are generally thread-safe for parallel processing

## Troubleshooting

**Common Issues:**

- **Import errors**: Ensure all dependencies are installed
- **Path not found**: Verify file paths and naming conventions
- **Memory errors**: Reduce batch sizes or image resolutions
- **Shape mismatches**: Check image dimensions consistency
- **Permission errors**: Verify write permissions for output directories

## Integration

The utils module integrates with all other components:
- **CropManager**: Uses cropping and file utilities
- **Data**: Uses image and data manipulation utilities
- **Model**: Uses model and evaluation utilities
- **Visualizer**: Uses plotting and image utilities
- **External Libraries**: Seamless integration with TensorFlow, OpenCV, PIL

---

The Utils module provides the foundational toolkit that enables the entire Cancer Segmentation framework, offering reliable, 
efficient, and well-tested utilities for medical image processing and analysis workflows.
