import os
import pandas as pd
from utils.image import is_valid_image_file, get_mask_path, load_image
import numpy as np


class DataBuilder:
    def __init__(self, root_dir, verbose=False):
        self.root_dir = root_dir
        self.verbose = verbose

    def build_dataframe(self):
        data = []

        for subfolder in os.listdir(self.root_dir):
            sub_path = os.path.join(self.root_dir, subfolder)
            if not os.path.isdir(sub_path):
                continue

            for file in os.listdir(sub_path):
                if not is_valid_image_file(file):
                    continue

                image_path = os.path.join(sub_path, file)
                mask_path = get_mask_path(image_path)

                if not os.path.exists(mask_path):
                    if self.verbose:
                        print(f"Missing mask for: {image_path}")
                    continue

                try:
                    mask_class = self.classify_mask(mask_path)
                    data.append((image_path, mask_path, mask_class))
                except Exception as e:
                    print(f"Could not process mask {mask_path}: {e}")

        return pd.DataFrame(data, columns=["image_path", "mask_path", "mask_class"])

    def classify_mask(self, mask_path):
        """Classify mask as positive or negative based on pixel values."""

        mask = load_image(mask_path, mode="L")
        mask_array = np.array(mask)
        return "negative" if np.sum(mask_array) == 0 else "positive"

