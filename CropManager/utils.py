import cv2


def extract_center_crops(image, crop_size=256):
    h, w = image.shape[:2]
    step_w = step_h = crop_size
    center_x, center_y = w // 2, h // 2
    offset = crop_size // 2


    centers = [(center_x - offset, center_y - offset)]

    for cx, cy in centers:
        # Clamp coordinates to image bounds
        cx = max(0, min(cx, w - step_w))
        cy = max(0, min(cy, h - step_h))

        # Extract crop
        crop_img = image[cy:cy + step_h, cx:cx + step_w]

        # Check if crop is valid
        if crop_img.shape[0] != step_h or crop_img.shape[1] != step_w:
            continue

        # Resize and store
        crop_img = cv2.resize(crop_img, (crop_size, crop_size))

    return crop_img
