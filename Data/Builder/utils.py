import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(df, train_ratio=0.8, test_ratio=0.2, seed=42):
    """Splits a DataFrame into training and testing sets."""
    assert abs((train_ratio + test_ratio) - 1.0) < 1e-6, "Train and test ratios must sum to 1."

    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed)

    print(f"Split Summary:\nTrain: {len(train_df)}\nTest: {len(test_df)}")

    return train_df, test_df


def sample_distribution(df, class_column, proportions, random_state=None):
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")
    if class_column not in df.columns:
        raise ValueError(f"Column '{class_column}' not found in DataFrame")
    if not isinstance(proportions, dict):
        raise ValueError("'proportions' must be a dictionary")
    if abs(sum(proportions.values()) - 1.0) > 1e-6:
        raise ValueError("Proportions must sum to 1.0")

    # Get class counts
    class_counts = df[class_column].value_counts()
    for cls in proportions:
        if cls not in class_counts:
            raise ValueError(f"Class '{cls}' not found in {class_column}")

    # Determine total_samples as the size of the smallest class multiplied by sum of proportions
    min_class_size = min(class_counts[cls] / prop for cls, prop in proportions.items() if prop > 0)
    total_samples = int(min_class_size * sum(proportions.values()))

    # Calculate sample sizes for each class
    sample_sizes = {cls: int(total_samples * prop) for cls, prop in proportions.items()}

    # Adjust for rounding errors to ensure total_samples is met
    total_assigned = sum(sample_sizes.values())
    if total_assigned != total_samples:
        largest_class = max(proportions, key=proportions.get)
        sample_sizes[largest_class] += total_samples - total_assigned

    # Verify sufficient data (should always be true due to min_class_size calculation)
    for cls, size in sample_sizes.items():
        available = class_counts.get(cls, 0)
        if size > available:
            print(f"Warning: Requested {size} samples for class '{cls}', but only {available} available. Adjusting to {available}.")
            sample_sizes[cls] = available

    # Sample from each class
    sampled_dfs = []
    for cls, size in sample_sizes.items():
        if size > 0:
            class_df = df[df[class_column] == cls].sample(n=size, random_state=random_state)
            sampled_dfs.append(class_df)

    # Combine and shuffle
    if not sampled_dfs:
        return pd.DataFrame(columns=df.columns)
    sampled_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Print sampled distribution
    print("\nSampled distribution:")
    print(sampled_df[class_column].value_counts())
    print(f"Total samples: {len(sampled_df)}")

    return sampled_df