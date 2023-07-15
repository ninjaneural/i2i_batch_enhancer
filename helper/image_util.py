import cv2
import numpy as np


def zoom_image(img, scale):
    interpolation = interpolation = cv2.INTER_CUBIC
    return cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=interpolation)


# resize|fit|crop
def resize_image(img, w, h, mode="resize"):
    (img_width, img_height) = img.shape[:2]
    if img_height + img_width < h + w:
        interpolation = interpolation = cv2.INTER_CUBIC
    else:
        interpolation = interpolation = cv2.INTER_AREA

    if mode == "crop":
        if w - img_width < h - img_height:
            scale = h / img_height
            x = (w - img_width) >> 1
            return cv2.resize(img, (img_width * scale, h), interpolation=interpolation)[0:h, x : x + w]
        elif w - img_width > h - img_height:
            scale = w / img_width
            y = (h - img_height) >> 1
            return cv2.resize(img, (w, img_height * scale), interpolation=interpolation)[y : y + h, 0:w]
    elif mode == "fit":
        if w - img_width > h - img_height:
            scale = h / img_height
            left = (w - img_width) >> 1
            right = (w - img_width) - left
            return np.pad(cv2.resize(img, (img_width * scale, h), interpolation=interpolation), pad_width=((0, 0), (left, right), (0, 0)), mode="constant")
        elif w - img_width < h - img_height:
            scale = w / img_width
            top = (h - img_height) >> 1
            bottom = (h - img_height) - top
            return np.pad(cv2.resize(img, (w, img_height * scale), interpolation=interpolation), pad_width=((top, bottom), (0, 0), (0, 0)), mode="constant")

    return cv2.resize(img, (w, h), interpolation=interpolation)


def crop_and_resize(img_array, coords, re_w, re_h, mode="resize"):
    (x, y, w, h) = coords
    crop_image = img_array[y : y + h, x : x + w]
    return resize_image(crop_image, re_w, re_h, mode)
