# CropManager

A Python module for intelligent image cropping and preprocessing, specifically designed for medical image segmentation tasks. The CropManager provides grid-based cropping strategies with center crop extraction to enhance model training and evaluation.

## Overview

CropManager implements a sophisticated cropping strategy that:
- Divides images into grid-based patches of various sizes
- Extracts center crops for focused analysis
- Automatically processes both images and their corresponding masks
- Organizes output in a structured directory format
- Supports batch processing of entire datasets

## Features

### Grid-Based Cropping
- **Multiple grid sizes**: Configurable grid dimensions (default: 4x4 to 10x10)
- **Uniform patch size**: All crops resized to consistent dimensions (default: 256x256)
- **Center crop extraction**: Additional center-focused crop for enhanced accuracy
- **Automatic resizing**: Intelligent resizing to maintain aspect ratio

### Mask Processing
- **Automatic mask detection**: Finds corresponding mask files (`*_mask.png`)
- **Synchronized cropping**: Applies identical cropping to images and masks
- **Missing mask handling**: Gracefully handles images without masks

### Batch Processing
- **Directory traversal**: Processes all images in subdirectories
- **Organized output**: Creates structured output directories per image
- **Progress tracking**: Built-in processing workflow


## Usage

### Basic Usage

```python
from CropManager import Crop

# Initialize the cropper
cropper = Crop(
    input_dir="path/to/input/images",
    output_dir="path/to/output/crops"
)

# Generate crops with default grid sizes (4x4 to 10x10)
cropper.generate_crops()
```

### Advanced Configuration

```python
# Custom grid sizes and output dimensions
cropper = Crop(
    input_dir="BUSI_DATASET",
    output_dir="CROPS"
)

# Generate crops with specific grid sizes
cropper.generate_crops(grid_sizes=range(3, 8))  # 3x3 to 7x7 grids

# Crop a single image with custom output size
crops = cropper.crop_image(
    image_path="sample_image.png",
    grid_size=5,
    crop_output=512  # 512x512 output size
)
```

### Complete Workflow Example

```python
import os
from CropManager import Crop

# Set up directories
input_directory = "raw_medical_images"
output_directory = "processed_crops"

# Initialize cropper
cropper = Crop(input_dir=input_directory, output_dir=output_directory)

# Process all images with multiple grid strategies
grid_configurations = [4, 5, 6, 7, 8, 9, 10]
success = cropper.generate_crops(grid_sizes=grid_configurations)

if success:
    print("✅ All images processed successfully!")
    print(f"📁 Crops saved to: {output_directory}")
```

## Input/Output Structure

### Expected Input Structure
```
input_dir/
├── subfolder1/
│   ├── image1.png
│   ├── image1_mask.png
│   ├── image2.png
│   └── image2_mask.png
├── subfolder2/
│   ├── image3.png
│   ├── image3_mask.png
│   └── ...
└── ...
```

### Generated Output Structure
```
output_dir/
├── image_0/                    # First processed image
│   ├── image_0_grid_4_0.png    # Grid crop 0 from 4x4 grid
│   ├── image_0_grid_4_0_mask.png
│   ├── image_0_grid_4_1.png    # Grid crop 1 from 4x4 grid
│   ├── image_0_grid_4_1_mask.png
│   ├── ...
│   ├── image_0_grid_4_16.png   # Center crop from 4x4 processing
│   ├── image_0_grid_4_16_mask.png
│   ├── image_0_grid_5_0.png    # 5x5 grid crops
│   └── ...
├── image_1/                    # Second processed image
│   └── ...
└── ...
```

## Class Reference

### `Crop`

Main class for image cropping operations.

#### Constructor
```python
Crop(input_dir=None, output_dir=None)
```

**Parameters:**
- `input_dir` (str): Directory containing input images
- `output_dir` (str): Directory for saving cropped images

#### Methods

##### `crop_image(image_path, grid_size, crop_output=256)`
Crops a single image using specified grid size.

**Parameters:**
- `image_path` (str): Path to the input image
- `grid_size` (int): Grid dimensions (n×n)
- `crop_output` (int): Output crop size in pixels

**Returns:**
- `list`: List of cropped image arrays

##### `generate_crops(grid_sizes=range(4, 11))`
Processes all images in input directory with multiple grid sizes.

**Parameters:**
- `grid_sizes` (iterable): Sequence of grid sizes to apply

**Returns:**
- `bool`: True if processing completed successfully

## Key Features Explained

### Grid-Based Strategy
The module divides each image into an n×n grid and extracts each cell as a separate crop:

```python
# For a 4x4 grid on a 1024x1024 image:
# - Creates 16 crops of 256x256 pixels each
# - Plus 1 center crop
# - Total: 17 crops per grid size per image
```

### Center Crop Enhancement
In addition to grid crops, extracts a center-focused crop:
- Improves focus on central image features
- Provides additional training data variety
- Maintains original aspect ratio when possible

### Automatic Mask Handling
- Detects mask files using `*_mask.png` naming convention
- Applies identical transformations to images and masks
- Ensures perfect alignment between input and ground truth

## ⚠Important Notes

1. **File Naming**: Mask files must follow the `*_mask.png` convention
2. **Image Format**: Currently supports PNG format
3. **Memory Usage**: Large datasets may require substantial memory for batch processing
4. **Directory Structure**: Creates organized subdirectories automatically

## Error Handling

The module includes robust error handling:
- Validates image file existence
- Handles missing mask files gracefully
- Provides informative error messages
- Continues processing even if individual images fail

## Integration

CropManager is designed to integrate seamlessly with:
- **Data pipelines**: Preprocessing step for ML workflows
- **TensorFlow/PyTorch**: Compatible output format for deep learning
- **Medical imaging**: Optimized for medical image characteristics
- **Batch processing**: Efficient handling of large datasets

## Performance Tips

1. **Batch Size**: Process images in batches for memory efficiency
2. **Grid Selection**: Start with 5x5 or 6x6 grids for optimal balance
3. **Output Size**: Use 256x256 for most deep learning applications
4. **Storage**: Ensure sufficient disk space for multiple grid sizes

## Troubleshooting

**Common Issues:**

- **"Image is not found"**: Verify file paths and permissions
- **Memory errors**: Reduce batch size or grid range
- **Missing masks**: Check file naming convention (`*_mask.png`)
- **Empty output**: Verify input directory contains valid PNG files

---

**Note**: This module is optimized for medical imaging applications but can be adapted for other computer vision tasks requiring image cropping strategies.
