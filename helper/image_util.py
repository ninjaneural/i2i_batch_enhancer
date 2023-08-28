import cv2
import numpy as np
from PIL import Image, ImageDraw


def zoom_image(img, scale):
    interpolation = interpolation = cv2.INTER_CUBIC
    return cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=interpolation)


# resize|fit|crop
def resize_image(img, w, h, mode="resize", anchor="center"):
    (img_height, img_width) = img.shape[:2]
    if img_height + img_width < h + w:
        interpolation = interpolation = cv2.INTER_CUBIC
    else:
        interpolation = interpolation = cv2.INTER_AREA

    if mode == "crop":
        if w / img_width < h / img_height:
            scale = h / img_height
            re_w = int(img_width * scale)
            re_h = h
            X = 0
            y = 0
            if anchor == "center":
                x = (re_w - w) >> 1
            elif anchor == "right":
                x = re_w - w
        else:
            scale = w / img_width
            re_w = w
            re_h = int(img_height * scale)
            x = 0
            y = 0
            if anchor == "center":
                y = (re_h - h) >> 1
            elif anchor == "bottom":
                y = re_h - h
        return cv2.resize(img, (re_w, re_h), interpolation=interpolation)[y : y + h, x : x + w]
    elif mode == "fit":
        if w / img_width > h / img_height:
            scale = h / img_height
            top = 0
            bottom = 0
            left = (w - img_width) >> 1
            right = (w - img_width) - left
            re_w = int(img_width * scale)
            re_h = h
        else:
            scale = w / img_width
            top = (h - img_height) >> 1
            bottom = (h - img_height) - top
            left = 0
            right = 0
            re_w = w
            re_h = int(img_height * scale)
        return np.pad(cv2.resize(img, (re_w, re_h), interpolation=interpolation), pad_width=((top, bottom), (left, right), (0, 0)), mode="constant")

    return cv2.resize(img, (w, h), interpolation=interpolation)


def crop_and_resize(img_array, coords, re_w, re_h, mode="resize"):
    (x, y, w, h) = coords
    crop_image = img_array[y : y + h, x : x + w]
    return resize_image(crop_image, re_w, re_h, mode)


def blur_masks(masks, dilation_factor, iter=1):
    dilated_masks = []
    if dilation_factor == 0:
        return masks
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
    for i in range(len(masks)):
        cv2_mask = masks[i]
        dilated_mask = cv2.erode(cv2_mask, kernel, iter)
        dilated_mask = cv2.GaussianBlur(dilated_mask, (51, 51), 0)

        dilated_masks.append(dilated_mask)
    return dilated_masks


def merge_image(bg_array, patch_array, coord, mask_array):
    (x, y, w, h) = coord
    print(f"coord {coord}")

    print(f"patch_array1 {patch_array.shape}")
    patch_array = resize_image(patch_array, w, h, "crop")
    mask_array = mask_array[y : y + h, x : x + w]
    mask_array = mask_array.astype(dtype="float") / 255
    if mask_array.ndim == 2:
        mask_array = mask_array[:, :, np.newaxis]

    bg = bg_array[y : y + h, x : x + w]
    print(f"bg {bg.shape}")
    print(f"mask_array {mask_array.shape}")
    print(f"patch_array {patch_array.shape}")
    # (p_h, p_w) = patch_array.shape[:2]
    patch_array = patch_array[0:h, 0:w]
    bg_array[y : y + h, x : x + w] = mask_array * patch_array + (1 - mask_array) * bg

    return bg_array
