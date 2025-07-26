# Cancer Segmentation

A deep learning framework for medical image segmentation using U-Net architecture with grid-based crop analysis for cancer detection and segmentation tasks.

## Overview

This project implements an end-to-end pipeline for cancer segmentation in medical images, featuring:

- **Grid-based cropping strategy** for increasing the number of trainig examples
- **U-Net architecture** with 1024 filters for precise medical image segmentation
- **Multi-grid analysis** to find optimal segmentation strategies
- **Comprehensive evaluation metrics** including IoU scoring
- **TensorFlow/Keras implementation** with TFRecord support
- **Interactive visualization tools** for results analysis


## Project Structure

```
Cancer-Segmentation/
├── CropManager/           # Image cropping and preprocessing utilities
│   ├── Crop.py           # Main cropping class with grid-based strategies
│   └── __init__.py
├── Data/                 # Data processing and TFRecord generation
│   ├── DataBuilder.py    # Dataset construction and validation
│   ├── TFRecordBuilder.py # TensorFlow record creation
│   └── __init__.py
├── Model/                # Deep learning models and inference
│   ├── Inference/        # Model evaluation and analysis tools
│   │   ├── Analyze.py    # Multi-image IoU analysis
│   │   ├── Evaluate.py   # Grid-based evaluation metrics
│   │   └── __init__.py
│   ├── zoo/              # Model architectures
│   │   ├── Unet_1024.py  # U-Net implementation
│   │   └── __init__.py
│   └── __init__.py
├── Notebooks/            # Jupyter notebooks for training and testing
│   ├── Training/         # Model training notebooks
│   ├── Testing/          # Model evaluation notebooks
│   └── Processing/       # Data preprocessing notebooks
├── Visualizer/           # Visualization and plotting utilities
│   ├── Grid.py           # Grid visualization tools
│   ├── Model.py          # Model prediction visualization
│   └── __init__.py
├── utils/                # Core utility functions
│   ├── crop.py           # Cropping algorithms
│   ├── data.py           # Data manipulation utilities
│   ├── file.py           # File I/O operations
│   ├── image.py          # Image processing functions
│   ├── model.py          # Model utilities and metrics
│   ├── plot.py           # Plotting and visualization
│   └── __init__.py
├── .gitignore
├── requirements.txt
├── LICENSE
└── main.py
```
### Additional Information for Each Module
- [CropManager](https://github.com/Nifdi01/Cancer-Segmentation/tree/main/CropManager)
- [Data](https://github.com/Nifdi01/Cancer-Segmentation/tree/main/Data)
- [Model](https://github.com/Nifdi01/Cancer-Segmentation/tree/main/Model)
- [Visualize](https://github.com/Nifdi01/Cancer-Segmentation/tree/main/Visualize)
- [Utilities](https://github.com/Nifdi01/Cancer-Segmentation/tree/main/utils)

## Features

### Grid-Based Segmentation Strategy
- **Multi-scale cropping**: Generates crops using various grid sizes (4x4 to 10x10)
- **Center crop extraction**: Additional center-focused crops for improved accuracy
- **Adaptive grid evaluation**: Automatically finds optimal grid size per image

### U-Net Architecture
- **Deep U-Net with 1024 filters**: Enhanced capacity for complex medical images
- **Batch normalization**: Improved training stability
- **Dropout regularization**: Prevents overfitting
- **Skip connections**: Preserves fine-grained details

### Comprehensive Evaluation
- **IoU (Intersection over Union)**: Primary segmentation metric
- **Multi-grid analysis**: Compares performance across different grid strategies
- **Statistical analysis**: Mean, std, min, max IoU scores per grid size
- **Visual evaluation**: Side-by-side comparison tools

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Cancer-Segmentation.git
cd Cancer-Segmentation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up data directories**:
```bash
mkdir -p BUSI_DATASET CROPS TFRecords Model/zoo/weights
```
4. **Download Models**

  Download the preferred model from [this link](https://drive.google.com/drive/folders/1opxv4jqbdBHxO5J0zAm7sRezhQ1QHfs3?usp=sharing) and put it inside `Model/zoo/weights` directory

5. **Run Notebooks**

  There are several examples inside `Notebooks` directory that showcase training, testing, and processing of model and data.

## Usage

### 1. Data Preparation

```python
from Data import DataBuilder, TFRecordBuilder
from utils.data import split_dataset

# Build dataset dataframe
builder = DataBuilder("path/to/your/dataset", verbose=True)
df = builder.build_dataframe()

# Split into train/test
train_df, test_df = split_dataset(df, train_ratio=0.8, test_ratio=0.2)

# Create TFRecords
train_writer = TFRecordBuilder(train_df, "TFRecords/")
train_writer.write_tfrecord("Train")

# Preprocess Data (see Notebooks/Processing/ for complete examples)
```

### 2. Image Cropping

```python
from CropManager import Crop

# Initialize cropper
cropper = Crop(input_dir="BUSI_DATASET", output_dir="CROPS")

# Generate crops with multiple grid sizes
cropper.generate_crops(grid_sizes=range(4, 11))

# Preprocess Data (see Notebooks/Processing/ for complete examples)
```

### 3. Model Training

```python
from Model.zoo import UNetModel1024

# Initialize model
model = UNetModel1024(input_shape=(256, 256, 1))
unet = model.load_model()

# Compile model
unet.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'iou']
)

# Train model (see Notebooks/Training/ for complete examples)
```

### 4. Model Evaluation

```python
from Model.Inference import analyze_iou_across_images
import numpy as np

# Load trained model
model = UNetModel1024(weights="Model/zoo/weights/best_model.h5")
unet = model.load_model()

# Analyze multiple images
image_indices = np.random.choice(range(100), 20, replace=False)
results = analyze_iou_across_images(
    output_dir="CROPS",
    model=unet,
    index_range=image_indices,
    mask_threshold=0.7,
    grid_range=(4, 10)
)

# Print summary statistics
from Model.Inference import get_summary
summary_df = get_summary(results)
print(summary_df)


# Test model (see Notebooks/Testing/ for complete examples)
```

### 5. Visualization

```python
from Visualizer import display_best_grid_prediction, display_grid_from_path

# Display best prediction for an image
best_grid, iou_scores = display_best_grid_prediction(
    output_dir="CROPS",
    image_idx=5,
    model=unet,
    mask_threshold=0.7
)

# Display original grid crops
display_grid_from_path("CROPS", image_idx=5, grid_size=6, mask=False)

# Test model (see Notebooks/Testing/ for complete examples)
```

## Model Performance

The framework evaluates segmentation performance using:

- **IoU Score**: Primary metric for segmentation accuracy
- **Grid Size Optimization**: Finds optimal cropping strategy per image
- **Statistical Analysis**: Comprehensive performance statistics
- **Visual Validation**: Side-by-side comparison of predictions vs ground truth

### Example Results
```
Grid Size Analysis Results:
├── 4x4 Grid: Mean IoU = 0.745 ± 0.082
├── 5x5 Grid: Mean IoU = 0.782 ± 0.067
├── 6x6 Grid: Mean IoU = 0.801 ± 0.054  ← Best
├── 7x7 Grid: Mean IoU = 0.788 ± 0.071
└── 8x8 Grid: Mean IoU = 0.763 ± 0.089
```

## Configuration

Key parameters can be adjusted in the respective modules:

- **Grid sizes**: Modify `grid_sizes` parameter (default: 4-10)
- **Crop output size**: Adjust `crop_output` parameter (default: 256x256)
- **Mask threshold**: Tune `mask_threshold` for binary predictions (default: 0.7)
- **Model architecture**: Currently, only one architecture is available on `Model/zoo/Unet_1024.py`

## Notebooks

The `Notebooks/` directory contains Jupyter notebooks for:

- **Training/**: Model training workflows and hyperparameter tuning
- **Testing/**: Model evaluation and performance analysis
- **Processing/**: Data preprocessing and augmentation techniques


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with TensorFlow/Keras for deep learning capabilities
- OpenCV for image processing operations
- Inspired by U-Net architecture for biomedical image segmentation
- Designed for medical imaging applications with emphasis on accuracy and interpretability

---

**Note**: This framework is designed for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.
