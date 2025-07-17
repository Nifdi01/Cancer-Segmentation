import os
import pandas as pd
import numpy as np
from PIL import Image

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
