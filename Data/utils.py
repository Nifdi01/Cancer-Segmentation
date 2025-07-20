import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import io


def load_image(image_path, mode="L"):
    """Load an image from the given path."""
    return Image.open(image_path).convert(mode)


def is_valid_image_file(filename):
    """Check if the file is a valid image (PNG, not a mask)."""
    return filename.endswith(".png") and not filename.endswith("_mask.png")


def get_mask_path(image_path):
    """Generate mask path from image path."""
    return image_path.replace(".png", "_mask.png")


def encode_image_to_bytes(img):
    """Encode PIL image to PNG bytes."""
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        return output.getvalue()


def split_dataset(df, train_ratio=0.8, test_ratio=0.2, seed=42):
    """Split a DataFrame into training and testing sets."""
    if abs((train_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Train and test ratios must sum to 1.")

    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed)
    print(f"Split Summary:\nTrain: {len(train_df)}\nTest: {len(test_df)}")
    return train_df, test_df


def sample_distribution(df, class_column, proportions, random_state=None):
    """Sample DataFrame based on class proportions."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")
    if class_column not in df.columns:
        raise ValueError(f"Column '{class_column}' not found in DataFrame")
    if not isinstance(proportions, dict):
        raise ValueError("'proportions' must be a dictionary")
    if abs(sum(proportions.values()) - 1.0) > 1e-6:
        raise ValueError("Proportions must sum to 1.0")

    class_counts = df[class_column].value_counts()
    for cls in proportions:
        if cls not in class_counts:
            raise ValueError(f"Class '{cls}' not found in {class_column}")

    min_class_size = min(class_counts[cls] / prop for cls, prop in proportions.items() if prop > 0)
    total_samples = int(min_class_size * sum(proportions.values()))
    sample_sizes = {cls: int(total_samples * prop) for cls, prop in proportions.items()}

    total_assigned = sum(sample_sizes.values())
    if total_assigned != total_samples:
        largest_class = max(proportions, key=proportions.get)
        sample_sizes[largest_class] += total_samples - total_assigned

    for cls, size in sample_sizes.items():
        available = class_counts.get(cls, 0)
        if size > available:
            print(
                f"Warning: Requested {size} samples for class '{cls}', but only {available} available. Adjusting to {available}.")
            sample_sizes[cls] = available

    sampled_dfs = [
        df[df[class_column] == cls].sample(n=size, random_state=random_state)
        for cls, size in sample_sizes.items() if size > 0
    ]

    if not sampled_dfs:
        return pd.DataFrame(columns=df.columns)

    sampled_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df