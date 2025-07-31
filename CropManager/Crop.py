import cv2
from utils.crop import create_grid_cells, extract_center_crops, resize_cells, save
from utils.image import crop_by_percentage
import os


class Crop:
    def __init__(self, input_dir=None, output_dir=None, filter=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filter = filter

    def crop_image(self, image_path, grid_size, crop_output=256, percentage=(0,0,0,0)):
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = crop_by_percentage(image, percentage)
        if image is None:
            raise ValueError("Image is not found or path is incorrect")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = self.filter.apply(image)

        # Create grid cells and append center crops
        grid_cells = create_grid_cells(image, grid_size)
        grid_cells.append(extract_center_crops(image, crop_output))

        # Resize all cells to the same size
        return resize_cells(grid_cells, crop_output)


    def generate_crops(self, grid_sizes=range(4, 11), crop_output=256, percentage=(0,0,0,0)):
        # Walk through all subdirectories and list all image files (excluding mask files)
        image_files = []
        for subdir, dirs, files in os.walk(self.input_dir):
            for f in files:
                if f.endswith('.png') and not f.endswith('_mask.png'):
                    image_files.append(os.path.join(subdir, f))

        for image_idx, image_path in enumerate(image_files[:1]):
            # Create a subdirectory for this image inside output_dir
            image_folder = os.path.join(self.output_dir, f"image_{image_idx}")
            os.makedirs(image_folder, exist_ok=True)

            # Construct mask path
            mask_path = image_path.replace('.png', '_mask.png')

            # Read mask if exists
            mask = None
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            for grid_size in grid_sizes:
                # CropManager image
                image_crops = self.crop_image(image_path, grid_size, crop_output, percentage)
                # CropManager mask if available
                mask_crops = []
                if mask is not None:
                    mask_crops = self.crop_image(mask_path, grid_size, crop_output, percentage)

                # Save crops in the image's subfolder
                save(image_crops, mask_crops, image_idx, grid_size, image_folder)

        return True