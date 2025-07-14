import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def build_dataframe(root_dir):
    data = []

    for subfolder in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                # An image ends with .png but NOT with mask.png
                if file.endswith(".png") and not file.endswith("_mask.png"):
                    image_path = os.path.join(sub_path, file)
                    # Construct mask path by inserting '_mask' before '.png'
                    mask_path = image_path.replace(".png", "_mask.png")

                    if os.path.exists(mask_path):
                        # Open mask and check if it's all black
                        try:
                            mask = Image.open(mask_path).convert("L")
                            mask_array = np.array(mask)

                            # Skip if the mask is entirely black (sum of pixel values is 0)
                            if np.sum(mask_array) == 0:
                                mask_class = "negative"
                            else:
                                mask_class = "positive"

                            data.append((image_path, mask_path, mask_class))
                        except Exception as e:
                            print(f"Could not process mask {mask_path}: {e}")
                    else:
                        # This message is useful for debugging missing files
                        print(f"Missing mask for: {image_path}")
                        pass

    df = pd.DataFrame(data, columns=["image_path", "mask_path", "mask_class"])
    return df


def split_dataset(df, train_ratio=0.8, test_ratio=0.2, seed=42):
    """Splits a DataFrame into training and testing sets."""
    assert abs((train_ratio + test_ratio) - 1.0) < 1e-6, "Train and test ratios must sum to 1."

    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed)

    print(f"Split Summary:\nTrain: {len(train_df)}\nTest: {len(test_df)}")

    return train_df, test_df
