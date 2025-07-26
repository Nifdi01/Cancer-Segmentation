# Data Module

The Data module provides comprehensive tools for dataset construction, validation, and TensorFlow record generation for medical image segmentation tasks.

## Overview

This module handles the entire data pipeline from raw medical images to training-ready TensorFlow records, including:
- Dataset structure validation and DataFrame construction
- Automatic mask classification (positive/negative)
- Efficient TensorFlow record serialization
- Progress tracking and error handling

## Module Structure

```
Data/
├── DataBuilder.py      # Dataset construction and validation
├── TFRecordBuilder.py  # TensorFlow record generation
└── __init__.py         # Module initialization
```

## Components

### DataBuilder

Constructs structured datasets from medical image directories with automatic validation and classification.

**Key Features:**
- Recursive directory traversal
- Automatic mask detection and pairing
- Binary classification (positive/negative) based on mask content
- Comprehensive error handling and logging
- DataFrame output for easy manipulation

### TFRecordBuilder

Converts pandas DataFrames to TensorFlow records for efficient training pipeline integration.

**Key Features:**
- Efficient binary serialization
- Progress tracking with tqdm
- Automatic train/validation/test splitting
- Memory-efficient processing
- PIL image encoding support

## Usage Examples

### Basic Dataset Construction

```python
from Data import DataBuilder

# Initialize builder with verbose logging
builder = DataBuilder(root_dir="BUSI_DATASET", verbose=True)

# Build comprehensive dataset DataFrame
df = builder.build_dataframe()

# Inspect the results
print(f"Total samples: {len(df)}")
print(f"Positive samples: {len(df[df['mask_class'] == 'positive'])}")
print(f"Negative samples: {len(df[df['mask_class'] == 'negative'])}")
print(df.head())
```

### TensorFlow Record Generation

```python
from Data import TFRecordBuilder
from utils.data import split_dataset

# Split dataset
train_df, test_df = split_dataset(df, train_ratio=0.8, test_ratio=0.2)

# Create training TFRecords
train_writer = TFRecordBuilder(train_df, tfrecord_path="TFRecords/")
train_writer.write_tfrecord("Train")

# Create testing TFRecords
test_writer = TFRecordBuilder(test_df, tfrecord_path="TFRecords/")
test_writer.write_tfrecord("Test")
```

### Complete Pipeline Workflow

```python
from Data import DataBuilder, TFRecordBuilder
from utils.data import split_dataset, sample_distribution
import os

# Ensure output directory exists
os.makedirs("TFRecords", exist_ok=True)

# Step 1: Build dataset
print("🔨 Building dataset...")
builder = DataBuilder("raw_images/", verbose=True)
df = builder.build_dataframe()

# Step 2: Balance dataset (optional)
balanced_df = sample_distribution(
    df, 
    class_column='mask_class',
    proportions={'positive': 0.6, 'negative': 0.4},
    random_state=42
)

# Step 3: Split dataset
train_df, test_df = split_dataset(balanced_df)

# Step 4: Create TFRecords
for split_name, split_df in [("Train", train_df), ("Test", test_df)]:
    print(f"📦 Creating {split_name} TFRecords...")
    writer = TFRecordBuilder(split_df, "TFRecords/")
    writer.write_tfrecord(split_name)

print("✅ Data pipeline completed!")
```

## API Reference

### DataBuilder Class

```python
DataBuilder(root_dir, verbose=False)
```

**Parameters:**
- `root_dir` (str): Root directory containing medical images
- `verbose` (bool): Enable detailed logging

**Methods:**

#### `build_dataframe()`
Constructs a pandas DataFrame from the image directory structure.

**Returns:**
- `pandas.DataFrame`: DataFrame with columns ['image_path', 'mask_path', 'mask_class']

#### `classify_mask(mask_path)`
Classifies a mask as positive or negative based on pixel content.

**Parameters:**
- `mask_path` (str): Path to the mask image

**Returns:**
- `str`: 'positive' if mask contains non-zero pixels, 'negative' otherwise

### TFRecordBuilder Class

```python
TFRecordBuilder(dataset_df, tfrecord_path=None)
```

**Parameters:**
- `dataset_df` (pandas.DataFrame): DataFrame containing image and mask paths
- `tfrecord_path` (str): Output directory for TFRecord files

**Methods:**

#### `write_tfrecord(log_title="TFRecord")`
Writes the dataset to a TensorFlow record file.

**Parameters:**
- `log_title` (str): Determines output filename ("Train", "Validation", "Test", or custom)

#### `serialize_example(image_path, mask_path)`
Serializes a single image-mask pair into TensorFlow Example format.

**Parameters:**
- `image_path` (str): Path to input image
- `mask_path` (str): Path to corresponding mask

**Returns:**
- `bytes`: Serialized TensorFlow Example

## Expected Input Structure

```
root_dir/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   ├── benign (2).png
│   ├── benign (2)_mask.png
│   └── ...
├── malignant/
│   ├── malignant (1).png
│   ├── malignant (1)_mask.png
│   ├── malignant (2).png
│   ├── malignant (2)_mask.png
│   └── ...
└── normal/
    ├── normal (1).png
    ├── normal (1)_mask.png
    └── ...
```

## Output Structure

### DataFrame Output
```
   image_path                    mask_path                     mask_class
0  /path/to/image1.png          /path/to/image1_mask.png      positive
1  /path/to/image2.png          /path/to/image2_mask.png      negative
2  /path/to/image3.png          /path/to/image3_mask.png      positive
```

### TFRecord Output
```
TFRecords/
├── train.tfrecord    # Training data
├── val.tfrecord      # Validation data (if created)
└── test.tfrecord     # Testing data
```

## Important Notes

1. **File Naming Convention**: Mask files must follow the `*_mask.png` pattern
2. **Image Format**: Currently supports PNG format
3. **Memory Considerations**: Large datasets may require chunked processing
4. **Path Validation**: All paths are validated before processing
5. **Error Handling**: Invalid images are logged and skipped

## Troubleshooting

**Common Issues:**

- **"Missing mask for image"**: Ensure mask files follow `*_mask.png` naming
- **Empty DataFrame**: Check directory structure and file permissions
- **TFRecord writing errors**: Verify output directory exists and is writable
- **Memory errors**: Process datasets in smaller batches

**Debug Tips:**

```python
# Enable verbose mode for detailed logging
builder = DataBuilder("dataset/", verbose=True)

# Check for missing masks
df = builder.build_dataframe()
print(f"Total valid pairs: {len(df)}")

# Verify class distribution
print(df['mask_class'].value_counts())
```

## Integration

This module integrates seamlessly with:
- **TensorFlow/Keras**: Direct TFRecord support
- **pandas**: DataFrame-based workflows
- **scikit-learn**: Data splitting utilities
- **PIL/OpenCV**: Image processing pipelines
- **Training scripts**: Ready-to-use data format

## Performance Tips

1. **Batch Processing**: Process images in batches for memory efficiency
2. **Path Caching**: Cache validated paths for repeated operations
3. **Parallel Processing**: Consider multiprocessing for large datasets
4. **Storage**: Use SSDs for faster I/O operations
5. **Memory Management**: Monitor memory usage with large datasets

---

This Data module provides a robust foundation for medical image segmentation data pipelines, ensuring data integrity and efficient processing for deep learning workflows.
