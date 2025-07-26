# Model Module

The Model module provides deep learning architectures and comprehensive inference tools for medical image segmentation, featuring U-Net implementation and advanced grid-based evaluation strategies.

## Overview

This module encompasses the complete modeling pipeline:
- **U-Net Architecture**: High-capacity segmentation model with 1024 filters
- **Grid-Based Inference**: Multi-scale evaluation for optimal segmentation
- **Performance Analysis**: Comprehensive IoU-based evaluation metrics
- **Batch Processing**: Efficient inference across multiple images

## Module Structure

```
Model/
├── Inference/              # Model evaluation and analysis tools
│   ├── Analyze.py         # Multi-image IoU analysis
│   ├── Evaluate.py        # Grid-based evaluation metrics
│   └── __init__.py
├── zoo/                   # Model architectures
│   ├── Unet_1024.py      # U-Net with 1024 filters
│   └── __init__.py
└── __init__.py
```

## Components

### Model Zoo

#### UNetModel1024
Advanced U-Net architecture optimized for medical image segmentation.

**Architecture Features:**
- **Deep Encoder-Decoder**: 5-level hierarchical feature extraction
- **1024 Filter Bridge**: Maximum capacity for complex pattern recognition
- **Skip Connections**: Preserves fine-grained spatial details
- **Batch Normalization**: Stable training and improved convergence
- **Dropout Regularization**: Prevents overfitting (10% dropout rate)
- **Sigmoid Output**: Probability-based segmentation masks

### Inference Module

#### Analyze.py
Comprehensive multi-image analysis with statistical evaluation.

**Key Features:**
- Multi-image IoU analysis across different grid sizes
- Statistical summaries (mean, std, min, max, median)
- Failed image tracking and error handling
- Progress monitoring and detailed logging

#### Evaluate.py
Core evaluation engine for grid-based segmentation assessment.

**Key Features:**
- Single and multi-grid evaluation
- IoU-based performance metrics
- Batch inference optimization
- File validation and error handling

## Usage Examples

### Model Initialization and Training

```python
from Model.zoo import UNetModel1024

# Initialize model with custom input shape
model = UNetModel1024(input_shape=(256, 256, 1))
unet = model.load_model()

# Compile for training
unet.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'iou']
)

# Display model architecture
unet.summary()

# Load pre-trained weights (optional)
model_with_weights = UNetModel1024(
    input_shape=(256, 256, 1),
    weights="Model/zoo/weights/best_model.h5"
)
```

### Single Image Grid Evaluation

```python
from Model.Inference import predict_and_evaluate_grid_sizes

# Evaluate different grid sizes for a single image
best_grid_size, iou_scores = predict_and_evaluate_grid_sizes(
    output_dir="CROPS",
    image_idx=5,
    model=unet,
    mask_threshold=0.7,
    grid_range=(4, 10)
)

print(f"Best grid size: {best_grid_size}x{best_grid_size}")
print("All IoU scores:", iou_scores)
```

### Multi-Image Analysis

```python
from Model.Inference import analyze_iou_across_images, get_summary
import numpy as np

# Select random sample of images
image_indices = np.random.choice(range(100), 20, replace=False)

# Comprehensive analysis across multiple images
results = analyze_iou_across_images(
    output_dir="CROPS",
    model=unet,
    index_range=image_indices,
    mask_threshold=0.7,
    grid_range=(4, 10)
)

# Generate statistical summary
summary_df = get_summary(results)
print(summary_df)

# Access detailed results
print(f"Processed: {results['processed_images']}/{results['total_images']}")
print(f"Failed images: {len(results['failed_images'])}")
print(f"Average best IoU: {np.mean(results['best_iou_scores']):.4f}")
```

### Complete Evaluation Workflow

```python
from Model.zoo import UNetModel1024
from Model.Inference import analyze_iou_across_images, get_summary
from utils.data import iou_to_dataframe
import numpy as np
import pandas as pd

# Load trained model
model = UNetModel1024(weights="path/to/trained_model.h5")
unet = model.load_model()

# Define evaluation parameters
evaluation_params = {
    'output_dir': "CROPS",
    'model': unet,
    'mask_threshold': 0.7,
    'grid_range': (4, 10)
}

# Sample images for evaluation
n_samples = 50
total_images = 200  # Adjust based on your dataset
image_indices = np.random.choice(range(total_images), n_samples, replace=False)

print(f"🔍 Evaluating {n_samples} images...")

# Run comprehensive analysis
results = analyze_iou_across_images(
    index_range=image_indices,
    **evaluation_params
)

# Generate reports
summary_df = get_summary(results)
detailed_df = iou_to_dataframe(results, image_indices)

# Display results
print("\n📊 Summary Statistics:")
print(summary_df.round(4))

print(f"\n🎯 Best Grid Size Distribution:")
best_grids = pd.Series(results['best_grid_sizes'])
print(best_grids.value_counts().sort_index())

print(f"\n📈 Performance Metrics:")
print(f"Average IoU: {np.mean(results['best_iou_scores']):.4f} ± {np.std(results['best_iou_scores']):.4f}")
print(f"Success Rate: {results['processed_images']}/{results['total_images']} ({100*results['processed_images']/results['total_images']:.1f}%)")
```

## API Reference

### UNetModel1024 Class

```python
UNetModel1024(input_shape=(256, 256, 1), weights=None)
```

**Parameters:**
- `input_shape` (tuple): Input image dimensions (height, width, channels)
- `weights` (str, optional): Path to pre-trained weights file

**Methods:**

#### `load_model()`
Returns the compiled Keras model.

**Returns:**
- `tensorflow.keras.Model`: Compiled U-Net model

#### Architecture Details:
```
Input (256, 256, 1)
    ↓
Encoder Block 1: 64 filters  → Skip Connection 1
    ↓ (MaxPool)
Encoder Block 2: 128 filters → Skip Connection 2
    ↓ (MaxPool)
Encoder Block 3: 256 filters → Skip Connection 3
    ↓ (MaxPool)
Encoder Block 4: 512 filters → Skip Connection 4
    ↓ (MaxPool)
Bridge: 1024 filters
    ↓ (UpSample)
Decoder Block 1: 512 filters ← Skip Connection 4
    ↓ (UpSample)
Decoder Block 2: 256 filters ← Skip Connection 3
    ↓ (UpSample)
Decoder Block 3: 128 filters ← Skip Connection 2
    ↓ (UpSample)
Decoder Block 4: 64 filters  ← Skip Connection 1
    ↓
Output: 1 filter (Sigmoid)
```

### Inference Functions

#### `predict_and_evaluate_grid_sizes()`
```python
predict_and_evaluate_grid_sizes(output_dir, image_idx, model, 
                               mask_threshold=0.5, grid_range=(4, 10))
```

**Parameters:**
- `output_dir` (str): Directory containing cropped images
- `image_idx` (int): Index of the image to evaluate
- `model`: Trained segmentation model
- `mask_threshold` (float): Threshold for binary predictions
- `grid_range` (tuple): Range of grid sizes to evaluate

**Returns:**
- `tuple`: (best_grid_size, iou_scores_dict)

#### `analyze_iou_across_images()`
```python
analyze_iou_across_images(output_dir, model, index_range, 
                         mask_threshold=0.7, grid_range=(4, 10))
```

**Parameters:**
- `output_dir` (str): Directory containing cropped images
- `model`: Trained segmentation model
- `index_range` (list): List of image indices to analyze
- `mask_threshold` (float): Threshold for binary predictions
- `grid_range` (tuple): Range of grid sizes to evaluate

**Returns:**
- `dict`: Comprehensive results dictionary with statistics

#### `get_summary(results)`
Generates statistical summary from analysis results.

**Parameters:**
- `results` (dict): Results from `analyze_iou_across_images()`

**Returns:**
- `pandas.DataFrame`: Summary statistics by grid size

## Performance Metrics

### IoU (Intersection over Union)
Primary segmentation metric calculated as:
```
IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
```

### Grid Size Optimization
The system evaluates multiple grid sizes to find the optimal strategy:
- **Small grids (4x4, 5x5)**: Capture global context
- **Medium grids (6x6, 7x7)**: Balance detail and context
- **Large grids (8x8, 9x9, 10x10)**: Focus on fine details

### Statistical Analysis
Comprehensive statistics include:
- **Mean IoU**: Average performance across images
- **Standard Deviation**: Performance consistency
- **Min/Max IoU**: Performance range
- **Median IoU**: Robust central tendency
- **Success Rate**: Percentage of successfully processed images

## Model Architecture Benefits

### U-Net with 1024 Filters
- **High Capacity**: 1024 filters in the bridge for complex pattern recognition
- **Medical Image Optimized**: Architecture proven effective for medical segmentation
- **Skip Connections**: Preserves fine-grained spatial information
- **Regularization**: Batch normalization and dropout prevent overfitting

### Grid-Based Evaluation
- **Multi-Scale Analysis**: Evaluates different receptive fields
- **Optimal Strategy Selection**: Finds best approach per image
- **Statistical Validation**: Robust performance assessment
- **Error Handling**: Graceful failure management

## Important Notes

1. **Memory Requirements**: 1024-filter U-Net requires substantial GPU memory
2. **Input Normalization**: Images should be normalized to [0, 1] range
3. **Batch Processing**: Use appropriate batch sizes for your hardware
4. **Weight Loading**: Ensure weight files match the model architecture
5. **Grid Crop Availability**: Requires pre-processed crops from CropManager

## Troubleshooting

**Common Issues:**

- **CUDA out of memory**: Reduce batch size or use smaller input resolution
- **Model loading errors**: Verify weight file compatibility
- **Low IoU scores**: Check data preprocessing and threshold values
- **Missing crop files**: Ensure CropManager has processed the images
- **Evaluation failures**: Validate file paths and image integrity

**Performance Optimization:**

```python
# Memory-efficient evaluation
import gc
import tensorflow as tf

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Clear memory between evaluations
def evaluate_with_cleanup(model, data):
    result = model.predict(data, batch_size=8)  # Smaller batch size
    gc.collect()  # Force garbage collection
    return result
```

## Integration

This module integrates with:
- **CropManager**: Requires pre-processed grid crops
- **Visualizer**: Provides data for visualization tools
- **TensorFlow/Keras**: Native TensorFlow model format
- **Training Pipelines**: Compatible with standard training workflows
- **Evaluation Scripts**: Seamless integration with analysis tools

---

The Model module provides state-of-the-art segmentation capabilities with comprehensive evaluation tools, enabling robust medical image analysis and research applications.
