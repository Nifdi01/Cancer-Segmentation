from PIL import Image
import cv2
import io


def load_image(image_path, mode="L"):
    """Load an image from the given path."""
    return Image.open(image_path).convert(mode)


def load_images(image_paths, mode=cv2.IMREAD_GRAYSCALE):
    """Load images from paths and validate they loaded successfully."""
    images = [cv2.imread(path, mode) for path in image_paths]

    if not images or any(img is None for img in images):
        return None

    return images


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


def crop_by_percentage(image, percentages):
    """
    Crop image by removing specified percentages from each side
    """

    h, w = image.shape[:2]
    top_pct, bottom_pct, left_pct, right_pct = percentages

    # Calculate pixels to remove from each side
    top = int(h * top_pct / 100)
    bottom = int(h * bottom_pct / 100)
    left = int(w * left_pct / 100)
    right = int(w * right_pct / 100)

    # Calculate crop boundaries
    y1 = top
    y2 = h - bottom
    x1 = left
    x2 = w - right

    # Crop the image
    return image[y1:y2, x1:x2]