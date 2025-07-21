import pandas as pd
from sklearn.model_selection import train_test_split


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


def iou_to_dataframe(results, index_range):
    """
    Create a more detailed DataFrame including all IoU scores by grid size.

    Args:
        results: Dictionary from analyze_iou_across_images function
        index_range: Original index range used in the analysis

    Returns:
        df: pandas DataFrame with detailed results
    """

    # Create list to store all records
    records = []

    # Get successful image indices (excluding failed ones)
    successful_indices = [idx for idx in index_range if idx not in results['failed_images']]

    # Method 2: Create DataFrame with multiple rows per image (one for each grid size)
    for i, image_idx in enumerate(successful_indices):
        best_iou = results['best_iou_scores'][i]
        best_grid = results['best_grid_sizes'][i]

        # Add rows for all grid sizes tested for this image
        for grid_size, iou_scores_list in results['all_iou_scores'].items():
            if i < len(iou_scores_list):  # Make sure this image has data for this grid size
                records.append({
                    'image_idx': image_idx,
                    'grid_size': grid_size,
                    'iou_score': iou_scores_list[i],
                    'is_best': grid_size == best_grid,
                    'best_iou_for_image': best_iou,
                    'best_grid_for_image': best_grid
                })

    df_detailed = pd.DataFrame(records)
    return df_detailed
